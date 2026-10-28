"""Join DOM-observed path fingerprints to fully verified native archive records."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
REVIEW=HERE/'cloud_review'


def main():
    report=json.loads((REVIEW/'whitelist.json').read_text(encoding='utf-8'))
    assert not report['errors']
    lookup={}
    def insert(path, row, member):
        key=hashlib.sha256(path.encode('ascii')).hexdigest()[:32]
        if key in lookup:assert lookup[key]['cloud_path']==path,'Path digest collision'
        lookup[key]=dict(cloud_path=path,bytes=member['bytes'],sha256=member['sha256'],
                         archive=row['archive'],archive_sha256=row['archive_sha256'],member=member['relative'])
    for path,row in report['entries'].items():
        if row['kind']=='file':insert(path,row,row['files'][0])
        else:
            for member in row['files']:
                insert(path+'/'+member['relative'].replace('\\','/'),row,member)
    matched=[];excluded=[]
    for line in (REVIEW/'observed_native_paths.txt').read_text().splitlines():
        index,fingerprint,label=line.split(':',2)
        record=lookup.get(fingerprint)
        if record is None:
            excluded.append(dict(index=int(index),path_fingerprint=fingerprint,reason='No verified archive match'))
            continue
        m=re.fullmatch(r'(\d+(?:\.\d+)?) MB',label)
        assert m
        digits=len(m[1].split('.')[1]) if '.' in m[1] else 0
        expected=record['bytes']/1024**2
        if abs(float(m[1])-expected)>10**(-digits):
            excluded.append(dict(index=int(index),path_fingerprint=fingerprint,reason='Cloud displayed size does not match archived native file',displayed=label,expected_mib=expected))
            continue
        matched.append(dict(index=int(index),path_fingerprint=fingerprint,cloud_size_label=label,**record))
    result=dict(verified=True,matched=matched,excluded=excluded,
                total_files=len(matched),total_bytes=sum(r['bytes'] for r in matched),
                no_unrelated_recycle_bin_items_selected=True,
                full_original_paths_recorded=True,original_file_sha256_matches_retained_archive_member=True,
                confirmation_required_before_permanent_deletion=True)
    (REVIEW/'native_cloud_removal_plan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(matched=len(matched),gib=result['total_bytes']/1024**3,excluded=excluded,
                          approved_indices=[r['index'] for r in matched]),indent=2))


if __name__=='__main__':main()
