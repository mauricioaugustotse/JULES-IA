"""Regressions from real DJe contamination of 2019/2024 session records."""
import pytest
from manutencao_sessoes.dje_identity import same_full_electoral_cnj


@pytest.mark.parametrize('session,document', [
    ('0600227-31.2018.6.17.0000', '0600227-31.2024.6.06.0031'),
    ('0600071-96.2021.6.14.0040', '0600071-96.2024.6.26.0286'),
    ('0600492-36.2020.6.00.0000', '0600492-36.2024.6.17.0028'),
    ('0600227-31', '0600227-31.2018.6.17.0000'),
    ('0600227-31.2018.6.17.0000', ''),
    ('', ''),
    ('0600227-31.2018.6.17.0000', '0600227-31.2018.6.17.0000; 0600071-96.2021.6.14.0040'),
    ('0600227-31.2018.8.17.0000', '0600227-31.2018.8.17.0000'),
])
def test_rejects_other_or_incomplete_identity(session, document):
    assert not same_full_electoral_cnj(session, document)


@pytest.mark.parametrize('session,document', [
    ('0600227-31.2018.6.17.0000', '06002273120186170000'),
    (' 0600071–96.2021.6.14.0040 ', '0600071-96.2021.6.14.0040'),
    # Preserve a source-confirmed historical identity despite its checksum.
    ('0000017-20.2012.6.19.0074', '0000017-20.2012.6.19.0074'),
])
def test_accepts_same_complete_source_identity(session, document):
    assert same_full_electoral_cnj(session, document)
