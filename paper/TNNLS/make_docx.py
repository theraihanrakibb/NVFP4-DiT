#!/usr/bin/env python3
"""Generate IEEE TNNLS submission .docx files (Title Page, COI, Cover Letter).

Uses only the Python standard library (zipfile + minimal OOXML) so it runs
without pandoc / LibreOffice on Windows.

NOTE on anonymization:
  * Main Manuscript (NVFP4-DiT_TNNLS.pdf) is fully anonymized and goes to
    reviewers.  The Title Page below is admin-only (NOT sent to reviewers),
    so real author names / affiliations are correct here.
  * Acknowledgments are intentionally OMITTED.  Do not add bracketed
    placeholders, fabricated grants, or "thank the reviewers" text.  Add a
    real acknowledgment only if the author supplies confirmed wording.
"""

import zipfile
import os

# ---------------------------------------------------------------------------
# OOXML helpers
# ---------------------------------------------------------------------------

def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def run(text, bold=False, italic=False, size=None):
    rpr = ''
    props = ''
    if bold:
        props += '<w:b/>'
    if italic:
        props += '<w:i/>'
    if size is not None:
        props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    if props:
        rpr = f'<w:rPr>{props}</w:rPr>'
    # preserve spaces
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def para(specs, after=120, before=0, style=None):
    ppr = '<w:pPr>'
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    ppr += f'<w:spacing w:before="{before}" w:after="{after}"/>'
    ppr += '</w:pPr>'
    runs = ''.join(specs)
    return f'<w:p>{ppr}{runs}</w:p>'


def build(paras, path):
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        + ''.join(paras)
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
          '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
          'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        '</Relationships>'
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:docDefaults><w:rPrDefault><w:rPr>'
        '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/>'
        '</w:rPr></w:rPrDefault></w:docDefaults>'
        '</w:styles>'
    )
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels)
        z.writestr('word/document.xml', document)
        z.writestr('word/_rels/document.xml.rels', doc_rels)
        z.writestr('word/styles.xml', styles)
    print(f'wrote {path} ({os.path.getsize(path)} bytes)')


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------

def title_page_paras():
    p = []
    p.append(para([run('NVFP4-DiT: Efficient 4-Bit Audio-Guided Video '
                       'Diffusion Transformers for Low-Cost Video Generation',
                       bold=True, size=28)]))
    p.append(para([run('Authors and Affiliations', bold=True, size=24)], before=120))
    p.append(para([run('Md Rakibul Islam Raihan', bold=True)]))
    p.append(para([run('School of Software, Northwestern Polytechnical University, '
                       "Xi'an, 710072, China")]))
    p.append(para([run('e-mail: raihan@mail.nwpu.edu.cn')]))
    p.append(para([run('Peng Zhang', bold=True)]))
    p.append(para([run('School of Computer Science, Northwestern Polytechnical '
                       'University, Xi\'an, 710072, China')]))
    p.append(para([run('e-mail: zh0036ng@nwpu.edu.cn')]))
    p.append(para([run('Corresponding Author', bold=True, size=24)], before=120))
    p.append(para([run('Md Rakibul Islam Raihan, School of Software, '
                       'Northwestern Polytechnical University, Xi\'an, 710072, China')]))
    p.append(para([run('e-mail: raihan@mail.nwpu.edu.cn')]))
    # Acknowledgments intentionally omitted -- no placeholders / fabricated text.
    return p


def coi_paras():
    p = []
    p.append(para([run('Conflict of Interest Statement', bold=True, size=26)]))
    p.append(para([run('None of the authors have any conflict of interest to '
                       'declare regarding the publication of this manuscript.')]))
    return p


def cover_letter_paras():
    p = []
    p.append(para([run('Cover Letter', bold=True, size=28)]))
    p.append(para([run('To the Editor-in-Chief of IEEE Transactions on Neural '
                       'Networks and Learning Systems,')], before=120))
    p.append(para([run('We are submitting the original research manuscript '
                       '"NVFP4-DiT: Efficient 4-Bit Audio-Guided Video Diffusion '
                       'Transformers for Low-Cost Video Generation" for consideration '
                       'of publication as a regular paper.')]))
    p.append(para([run('This manuscript is being submitted exclusively to IEEE '
                       'TNNLS and is not under consideration elsewhere. The work '
                       'has not been published previously.')]))
    p.append(para([run('In accordance with the IEEE TNNLS double-blind review '
                       'policy, the main manuscript has been fully anonymized: '
                       'author names, affiliations, acknowledgments, and '
                       'biographical information have been removed from the '
                       'manuscript that is sent to reviewers. The identifying '
                       'details are provided only in the separate Title Page.')]))
    p.append(para([run('We confirm that all authors have approved the manuscript '
                       'and agree with its submission to IEEE TNNLS.')]))
    p.append(para([run('Sincerely,')], before=120))
    p.append(para([run('Md Rakibul Islam Raihan (Corresponding Author)')]))
    p.append(para([run('School of Software, Northwestern Polytechnical University, '
                       "Xi'an, 710072, China")]))
    p.append(para([run('e-mail: raihan@mail.nwpu.edu.cn')]))
    return p


if __name__ == '__main__':
    build(title_page_paras(), 'TNNLS_TitlePage.docx')
    build(coi_paras(), 'TNNLS_ConflictOfInterest.docx')
    build(cover_letter_paras(), 'TNNLS_CoverLetter.docx')
