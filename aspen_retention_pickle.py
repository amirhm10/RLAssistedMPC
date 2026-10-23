"""Inspect checkpoint metadata without importing learners or executing pickle code.

Large NumPy/Torch byte payloads are skipped with seek. This is deliberately a
small, non-executing pickle interpreter: reducers/classes remain opaque, and
only recovery metadata and SHA-256 strings survive dictionary construction.
"""
import pickletools
import re


class InspectionError(ValueError):
    pass


FIELDS = frozenset(('schema', 'completed_steps', 'source_hashes', 'runner_sha256',
                    'native', 'file', 'file_sha256', 'project_file',
                    'project_file_sha256', 'name', 'live_save_unchanged'))
SHA256 = re.compile(r'^[0-9a-f]{64}$')
OPAQUE = object()
MARK = object()


def metadata(stream, *, max_memo=400_000, max_string=1_048_576):
    """Return root recovery fields, refusing truncation and unsupported formats."""
    start = stream.tell()
    stream.seek(0, 2)
    end = stream.tell()
    stream.seek(start)
    stack, memo = [], {}
    private_roots = set()
    protocol = None

    def exact(n):
        if n < 0 or stream.tell() + n > end:
            raise InspectionError('Truncated pickle argument')
        data = stream.read(n)
        if len(data) != n:
            raise InspectionError('Short pickle read')
        return data

    def items():
        try:
            index = len(stack) - 1 - stack[::-1].index(MARK)
        except ValueError as error:
            raise InspectionError('Missing pickle MARK') from error
        result = stack[index + 1:]
        del stack[index:]
        return result

    def assign(target, values):
        if len(values) % 2:
            raise InspectionError('Odd dictionary item count')
        if isinstance(target, dict):
            for key, value in zip(values[::2], values[1::2]):
                if key == '_e1_project_root' and isinstance(value, str):
                    private_roots.add(value)
                if isinstance(key, str) and (key in FIELDS or
                        isinstance(value, str) and SHA256.fullmatch(value)):
                    target[key] = value

    class Arguments:
        def read(self, size=-1):
            if not 0 <= size <= max_string:
                raise InspectionError('Oversized opcode argument')
            return exact(size)

        def readline(self):
            value = stream.readline(max_string + 1)
            if len(value) > max_string:
                raise InspectionError('Oversized opcode line')
            return value

    arguments = Arguments()

    while stream.tell() < end:
        code = exact(1).decode('latin1')
        info = pickletools.code2op.get(code)
        if info is None:
            raise InspectionError('Unknown pickle opcode')
        name = info.name
        if name in ('BINBYTES', 'BINBYTES8', 'SHORT_BINBYTES', 'BYTEARRAY8',
                    'BINUNICODE', 'BINUNICODE8', 'SHORT_BINUNICODE'):
            width = 1 if name.startswith('SHORT_') else 8 if name.endswith('8') else 4
            n = int.from_bytes(exact(width), 'little')
            if stream.tell() + n > end:
                raise InspectionError('Truncated binary payload')
            if 'UNICODE' in name:
                if n > max_string:
                    raise InspectionError('Oversized metadata string')
                arg = exact(n).decode('utf-8', 'surrogatepass')
            else:
                stream.seek(n, 1)
                arg = OPAQUE
        else:
            arg = info.arg.reader(arguments) if info.arg else None

        if name == 'PROTO':
            if protocol is not None or arg not in (4, 5):
                raise InspectionError('Only self-contained protocol 4/5 checkpoints are supported')
            protocol = arg
        elif protocol is None:
            raise InspectionError('Checkpoint has no supported protocol header')
        elif name == 'FRAME':
            if stream.tell() + arg > end:
                raise InspectionError('Truncated pickle frame')
        elif name == 'MARK':
            stack.append(MARK)
        elif name == 'EMPTY_DICT':
            stack.append({})
        elif name in ('EMPTY_LIST', 'EMPTY_TUPLE', 'EMPTY_SET'):
            stack.append(OPAQUE)
        elif name == 'NONE':
            stack.append(None)
        elif name in ('NEWTRUE', 'NEWFALSE'):
            stack.append(name == 'NEWTRUE')
        elif name in ('BININT', 'BININT1', 'BININT2', 'INT', 'LONG', 'LONG1', 'LONG4',
                      'FLOAT', 'BINFLOAT', 'BINUNICODE', 'BINUNICODE8', 'SHORT_BINUNICODE',
                      'BINBYTES', 'BINBYTES8', 'SHORT_BINBYTES', 'BYTEARRAY8',
                      'STRING', 'BINSTRING', 'SHORT_BINSTRING', 'UNICODE'):
            stack.append(arg)
        elif name in ('MEMOIZE', 'PUT', 'BINPUT', 'LONG_BINPUT'):
            key = len(memo) if name == 'MEMOIZE' else int(arg)
            if key > max_memo or len(memo) >= max_memo or not stack:
                raise InspectionError('Checkpoint memo limit exceeded')
            memo[key] = stack[-1]
        elif name in ('GET', 'BINGET', 'LONG_BINGET'):
            if int(arg) not in memo:
                raise InspectionError('Invalid pickle memo reference')
            stack.append(memo[int(arg)])
        elif name == 'SETITEM':
            value, key = stack.pop(), stack.pop()
            assign(stack[-1], [key, value])
        elif name == 'SETITEMS':
            values = items()
            assign(stack[-1], values)
        elif name == 'DICT':
            values = items()
            result = {}
            assign(result, values)
            stack.append(result)
        elif name in ('LIST', 'TUPLE', 'FROZENSET'):
            items()
            stack.append(OPAQUE)
        elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
            count = int(name[-1])
            if len(stack) < count:
                raise InspectionError('Invalid tuple stack')
            del stack[-count:]
            stack.append(OPAQUE)
        elif name == 'APPEND':
            stack.pop()
        elif name in ('APPENDS', 'ADDITEMS'):
            items()
        elif name in ('GLOBAL', 'EXT1', 'EXT2', 'EXT4', 'PERSID'):
            stack.append(OPAQUE)
        elif name in ('STACK_GLOBAL', 'REDUCE', 'NEWOBJ'):
            del stack[-2:]
            stack.append(OPAQUE)
        elif name == 'NEWOBJ_EX':
            del stack[-3:]
            stack.append(OPAQUE)
        elif name == 'BUILD':
            stack.pop()
        elif name == 'BINPERSID':
            stack[-1] = OPAQUE
        elif name == 'READONLY_BUFFER':
            pass
        elif name == 'DUP':
            stack.append(stack[-1])
        elif name == 'POP':
            stack.pop()
        elif name == 'POP_MARK':
            items()
        elif name == 'STOP':
            if stream.tell() != end or len(stack) != 1 or not isinstance(stack[0], dict):
                raise InspectionError('Checkpoint is not one complete root dictionary')
            result = stack[0]
            if result.get('schema') != 2 or not isinstance(result.get('completed_steps'), int):
                raise InspectionError('Not a schema-2 native training checkpoint')
            if not isinstance(result.get('native'), dict):
                raise InspectionError('Native checkpoint metadata is absent')
            result['retention_private_roots'] = sorted(private_roots)
            return result
        else:
            raise InspectionError('Unsupported pickle opcode: ' + name)
        if len(stack) > 100_000:
            raise InspectionError('Checkpoint stack limit exceeded')
    raise InspectionError('Checkpoint has no STOP opcode')
