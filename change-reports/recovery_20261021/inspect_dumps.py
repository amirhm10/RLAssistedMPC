"""Extract exception/module/register evidence from the three local crash dumps."""
import json
import mmap
from pathlib import Path
import struct

OUT = Path(__file__).parent
results = []
for pid in (22424, 23064, 45524):
    path = Path.home() / 'AppData/Local/CrashDumps' / f'python.exe.{pid}.dmp'
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        count, directory = struct.unpack_from('<II', data, 8)
        entries = {kind: (size, rva) for kind, size, rva in
                   [struct.unpack_from('<III', data, directory+i*12) for i in range(count)]}
        offset = entries[6][1]
        thread, = struct.unpack_from('<I', data, offset)
        code, flags, chain, address, number = struct.unpack_from('<IIQQI', data, offset+8)
        parameters = struct.unpack_from('<15Q', data, offset+40)[:number]
        context_size, context = struct.unpack_from('<II', data, offset+160)
        registers = dict(zip(('rax','rcx','rdx','rbx','rsp','rbp','rsi','rdi','r8','r9','r10','r11','r12','r13','r14','r15','rip'),
                            struct.unpack_from('<17Q', data, context+120)))
        modules = []
        module_list = entries[4][1]
        module_count, = struct.unpack_from('<I', data, module_list)
        for i in range(module_count):
            base, size, checksum, timestamp, name = struct.unpack_from('<QIIII', data, module_list+4+i*108)
            length, = struct.unpack_from('<I', data, name)
            filename = data[name+4:name+4+length].decode('utf-16-le')
            if base <= address < base+size or 'openblas' in filename.lower():
                modules.append(dict(base=hex(base), size=size, timestamp=hex(timestamp), path=filename,
                                    fault_rva=hex(address-base) if base <= address < base+size else None))
        instruction = None
        if 9 in entries:
            memory = entries[9][1]
            ranges, start = struct.unpack_from('<QQ', data, memory)
            for i in range(ranges):
                base, size = struct.unpack_from('<QQ', data, memory+16+i*16)
                if base <= address < base+size:
                    instruction = data[start+address-base:start+address-base+32].hex()
                start += size
        elif 5 in entries:
            memory = entries[5][1]
            ranges, = struct.unpack_from('<I', data, memory)
            for i in range(ranges):
                base, size, start = struct.unpack_from('<QII', data, memory+4+i*16)
                if base <= address < base+size:
                    instruction = data[start+address-base:start+address-base+32].hex()
        results.append(dict(pid=pid, thread_id=thread, exception=hex(code), parameters=[hex(v) for v in parameters],
                            exception_address=hex(address), registers={k:hex(v) for k,v in registers.items()},
                            instruction_hex=instruction, modules=modules))
(OUT/'crash_context.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
print(json.dumps(results, indent=2))
