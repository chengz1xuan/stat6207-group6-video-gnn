"""Check the delivered PDF, not just LaTeX declarations, for the requested IEEE styling."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.project-deps'))
import pymupdf


def main():
    config=json.loads((R/'artifacts/final_v4.json').read_text());path=R/config['report_path'];doc=pymupdf.open(path)
    assert len(doc)==4 and all(abs(p.rect.width-595.28)<.1 and abs(p.rect.height-841.89)<.1 for p in doc)
    page=doc[0];abstract=page.search_for('Abstract')[0];keywords=page.search_for('Index Terms')[0]
    spans=[s for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']]
    abstract_spans=[s for s in spans if s['text'].strip() and s['bbox'][0]<page.rect.width/2-4
                    and s['bbox'][1]>=abstract.y0-.5 and s['bbox'][3]<=keywords.y0+.5]
    assert len(abstract_spans)>20
    assert all(s['flags'] & 16 for s in abstract_spans),[(s['text'],s['font']) for s in abstract_spans if not s['flags']&16]
    authors=json.loads((R/'AUTHORS.json').read_text());rectangles=[]
    for author in authors:
        name=page.search_for(author.get('ieee_name',author['name']))[0];email=page.search_for(author['email'])[0]
        assert abs((name.x0+name.x1-email.x0-email.x1)/2)<3
        assert 15<email.y0-name.y0<60 and email.y1<abstract.y0
        rectangles.append(name)
    assert abs(rectangles[0].y0-rectangles[1].y0)<2 and abs(rectangles[2].y0-rectangles[3].y0)<2
    assert rectangles[2].y0>rectangles[0].y1 and rectangles[0].x0<rectangles[1].x0
    assert abs((rectangles[0].x0+rectangles[0].x1-rectangles[2].x0-rectangles[2].x1)/2)<3
    expected={
        'https://papers.nips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html',
        'https://arxiv.org/abs/1903.02428',
        'https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.SAGEConv.html',
        'https://www.ieee.org/conferences/publishing/templates.html',
    }
    links={link['uri'] for p in doc for link in p.get_links() if link.get('uri')}
    assert expected<=links
    text='\n'.join(p.get_text() for p in doc)
    assert all(value in text for value in ['1024','1034','Fey','Lenssen','1903.02428','SAGEConv'])
    xrefs={font[0] for p in doc for font in p.get_fonts()}
    assert all(doc.extract_font(xref)[3] for xref in xrefs)
    result={'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':len(doc),'A4':True,
            'abstract_all_text_spans_bold':True,'abstract_checked_spans':len(abstract_spans),
            'individual_IEEE_author_blocks':4,'author_rows':2,'email_identity_alignment':True,
            'given_name_family_name_order':True,'verified_reference_entries':4,'specific_reference_links_present':True,
            'all_fonts_embedded':True,'font_families':sorted({font[3] for p in doc for font in p.get_fonts()})}
    folder=R/config['primary_folder'];(folder/'report_format_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS', {k:v for k,v in result.items() if k!='font_families'})


if __name__=='__main__':main()
