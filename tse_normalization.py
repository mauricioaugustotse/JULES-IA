from __future__ import annotations

import ast
import csv
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlencode, urlparse


CNJ_REGEX = r"\b\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b"
SHORT_PROCESSO_REGEX = r"\b\d{3,7}-\d{2}\b"
LABELED_PROCESSO_REGEX = r"(?i)\bn(?:[º°]|\.)\s*(\d{3,7})\b"
SPECIAL_PROCESSO_REGEX = r"(?i)\b(?P<label>ADI|ADO)\s*(?P<number>\d{1,5})\b"
LABELED_SHORT_PROCESSO_REGEX = r"(?i)\b(?:re?sp(?:e)?|arespe|aresp|rhc|rms|ms|ro|pc)\s*(\d{5,9})\b"

CANON_CSV_FILENAME = "padrões para canonizar.csv"
CANON_DATA: Optional[dict[str, Any]] = None

CLASSE_PROCESSO_MAP = [
    (r"\bembargos de declaracao\b.*\b(agravo regimental|agravo interno|agrg|agr)\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "ED-AgRg-AREspe"),
    (r"\bed\s+agrg\s+arespe\b", "ED-AgRg-AREspe"),
    (r"\bembargos de declaracao\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "ED-AREspe"),
    (r"\bembargos de declaracao\b.*\b(recurso especial eleitoral|respe)\b", "ED-REspe"),
    (r"\bembargos de declaracao\b.*\b(recurso ordinario|ro)\b", "ED-RO"),
    (r"\bembargos de declaracao\b.*\b(prestacao de contas|pc)\b", "ED-PC"),
    (r"\bembargos de declaracao\b.*\b(lista triplice|lt)\b", "ED-Lista Tríplice"),
    (r"\bed\s+lt\b", "ED-Lista Tríplice"),
    (r"\b(agravo regimental|agravo interno|agrg|agr)\b.*\b(recurso especial eleitoral|respe)\b", "AgRg-REspe"),
    (r"\b(agravo regimental|agravo interno|agrg|agr)\b.*\b(agravo em recurso especial eleitoral|arespe)\b", "AgRg-AREspe"),
    (r"\b(agravo regimental|agravo interno|agrg|agr)\b.*\b(recurso ordinario|ro)\b", "AgRg-RO"),
    (r"\b(agravo regimental|agravo interno|agrg|agr)\b.*\b(mandado de seguranca|ms)\b", "AgRg-MS"),
    (r"\b(agravo regimental|agravo interno|agrg|agr)\b.*\b(prestacao de contas|pc)\b", "AgRg-PC"),
    (r"\breferend\w*\b.*\b(tutela cautelar antecedente|tutcautant)\b", "Ref-TutCautAnt"),
    (r"\bref\s*tutcautant\b", "Ref-TutCautAnt"),
    (r"\breferend\w*\b.*\b(mandado de seguranca|ms)\b", "Ref.-MS"),
    (r"\bref\s*ms\b", "Ref.-MS"),
    (r"\btutela cautelar antecedente\b|\btutcautant\b", "TutCautAnt"),
    (r"\blista triplice\b|\blt\b", "Lista Tríplice"),
    (r"\bprocesso administrativo\b|\bpa\b", "PA"),
    (r"\b(?:requisi[cç][aã]o|homologa[cç][aã]o).*for[cç]a federal\b|\bfor[cç]a federal\b", "PA"),
    (r"\bprestacao de contas\b|\bpc\b", "PC"),
    (r"\bconsulta\b|\bcta\b", "CTA"),
    (r"\bquestao de ordem\b|\bqo\b", "QO"),
    (r"\bpeticao civel\b|\bpetciv\b", "PetCiv"),
    (r"\ba[cç][aã]o direta de inconstitucionalidade por omiss[aã]o\b|\bado\b", "ADO"),
    (r"\ba[cç][aã]o direta de inconstitucionalidade\b|\badi\b", "ADI"),
    (r"\brecurso especial eleitoral\b|\brespe\b", "REspe"),
    (r"\bagravo em recurso especial eleitoral\b|\barespe\b", "AREspe"),
    (r"\brecurso ordinario\b|\bro\b", "RO"),
    (r"\brecurso em habeas corpus\b|\brhc\b", "RHC"),
    (r"\brecurso em mandado de seguranca\b|\brms\b", "RMS"),
    (r"\bmandado de seguranca\b|\bms\b", "MS"),
    (r"\bregistro\s+de\s+partido\s+pol[ií]tico\b", "RPP"),
    (r"\balter[aã]c[aã]o\s+(?:do|de|no)\s+registro\s+de\s+partido(?:\s+pol[ií]tico)?\b", "RPP"),
    (r"\balter[aã]c[aã]o\s+de\s+esta?tuto(?:\s+partid[aá]rio)?\b", "RPP"),
    (r"\brevis[aã]o\s+do\s+eleitorado\b", "RvE"),
    (r"\brpp\b", "RPP"),
    (r"\brve\b", "RvE"),
]

STATE_UF = {
    "acre": "AC",
    "alagoas": "AL",
    "amapá": "AP",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceará": "CE",
    "ceara": "CE",
    "distrito federal": "DF",
    "espírito santo": "ES",
    "espirito santo": "ES",
    "goiás": "GO",
    "goias": "GO",
    "maranhão": "MA",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "pará": "PA",
    "para": "PA",
    "paraíba": "PB",
    "paraiba": "PB",
    "paraná": "PR",
    "parana": "PR",
    "pernambuco": "PE",
    "piauí": "PI",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondônia": "RO",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "são paulo": "SP",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}
STATE_NAME_KEYS = set(STATE_UF.keys())
UF_CAPITALS = {
    "AC": "Rio Branco/AC",
    "AL": "Maceió/AL",
    "AP": "Macapá/AP",
    "AM": "Manaus/AM",
    "BA": "Salvador/BA",
    "CE": "Fortaleza/CE",
    "DF": "Brasília/DF",
    "ES": "Vitória/ES",
    "GO": "Goiânia/GO",
    "MA": "São Luís/MA",
    "MT": "Cuiabá/MT",
    "MS": "Campo Grande/MS",
    "MG": "Belo Horizonte/MG",
    "PA": "Belém/PA",
    "PB": "João Pessoa/PB",
    "PR": "Curitiba/PR",
    "PE": "Recife/PE",
    "PI": "Teresina/PI",
    "RJ": "Rio de Janeiro/RJ",
    "RN": "Natal/RN",
    "RS": "Porto Alegre/RS",
    "RO": "Porto Velho/RO",
    "RR": "Boa Vista/RR",
    "SC": "Florianópolis/SC",
    "SP": "São Paulo/SP",
    "SE": "Aracaju/SE",
    "TO": "Palmas/TO",
}


def _early_normalize_class_text(value: str) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFD", str(value).lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


CAPITAL_NAME_TO_VALUE = {
    _early_normalize_class_text(value.rsplit("/", 1)[0]): value
    for value in UF_CAPITALS.values()
}

MINISTRO_ALIAS_MAP = {
    "antonio carlos ferreira": "Min. Antônio Carlos Ferreira",
    "alexandre de moraes": "Min. Alexandre de Moraes",
    "andre mendonca": "Min. André Mendonça",
    "andre luiz mendonca": "Min. André Mendonça",
    "stella aranha": "Min. Estela Aranha",
    "antonio carlos": "Min. Antônio Carlos Ferreira",
    "admar gonzaga": "Min. Admar Gonzaga",
    "amar gonzaga": "Min. Admar Gonzaga",
    "arnaldo versiani": "Min. Arnaldo Versiani",
    "maria isabel gallotti": "Min. Isabel Gallotti",
    "isabel gallotti": "Min. Isabel Gallotti",
    "benedito goncalves": "Min. Benedito Gonçalves",
    "andre ramos tavares": "Min. Ramos Tavares",
    "ramos tavares": "Min. Ramos Tavares",
    "carlos mario da silva velloso filho": "Min. Carlos Mário da Silva Velloso Filho",
    "carlos bastide horbach": "Min. Carlos Horbach",
    "carlos horbach": "Min. Carlos Horbach",
    "carmen lucia": "Min. Cármen Lúcia",
    "cristiano zanin": "Min. Cristiano Zanin",
    "dias toffoli": "Min. Dias Toffoli",
    "edson fachin": "Min. Edson Fachin",
    "estela aranha": "Min. Estela Aranha",
    "flavio dino": "Min. Flávio Dino",
    "floriano de azevedo marques": "Min. Floriano de Azevedo Marques",
    "floriano de azevedo marques neto": "Min. Floriano de Azevedo Marques",
    "floriano marques": "Min. Floriano de Azevedo Marques",
    "gilmar mendes": "Min. Gilmar Mendes",
    "herman benjamin": "Min. Herman Benjamin",
    "henrique neves": "Min. Henrique Neves da Silva",
    "henrique neves da silva": "Min. Henrique Neves da Silva",
    "humberto jacques de medeiros": "Min. Humberto Jacques de Medeiros",
    "humberto martins": "Min. Humberto Martins",
    "joao otavio de noronha": "Min. João Otávio de Noronha",
    "joelson dias": "Min. Joelson Dias",
    "jorge mussi": "Min. Jorge Mussi",
    "laurita vaz": "Min. Laurita Vaz",
    "luciana lossio": "Min. Luciana Lóssio",
    "carmen lossio": "Min. Luciana Lóssio",  # transcricao errada de audio antigo (so ha 1 Lossio no TSE)
    "luis edson fachin": "Min. Edson Fachin",
    "luiz edson fachin": "Min. Edson Fachin",
    "luis felipe salomao": "Min. Luís Felipe Salomão",
    "luiz felipe salomao": "Min. Luís Felipe Salomão",
    "luis salomao": "Min. Luís Felipe Salomão",
    "luiz salomao": "Min. Luís Felipe Salomão",
    "luis fux": "Min. Luiz Fux",
    "luiz fux": "Min. Luiz Fux",
    "luis roberto barroso": "Min. Luís Roberto Barroso",
    "luiz roberto barroso": "Min. Luís Roberto Barroso",
    "roberto barroso": "Min. Luís Roberto Barroso",
    "marco aurelio": "Min. Marco Aurélio",
    "maria claudia bucchianeri pinheiro": "Min. Maria Cláudia Bucchianeri",
    "maria thereza": "Min. Maria Thereza de Assis Moura",
    "maria thereza de assis moura": "Min. Maria Thereza de Assis Moura",
    "mauro campbell": "Min. Mauro Campbell Marques",
    "mauro campbell marques": "Min. Mauro Campbell Marques",
    "napoleao maia": "Min. Napoleão Nunes Maia Filho",
    "napoleao nunes maia": "Min. Napoleão Nunes Maia Filho",
    "napoleao nunes maia filho": "Min. Napoleão Nunes Maia Filho",
    "luciano nunes maia": "Min. Napoleão Nunes Maia Filho",  # transcricao errada de audio antigo
    "luciano nunes maia filho": "Min. Napoleão Nunes Maia Filho",
    "nunes marques": "Min. Nunes Marques",
    "og fernandes": "Min. Og Fernandes",
    "paulo de tarso sanseverino": "Min. Paulo de Tarso Sanseverino",
    "paulo tarso sanseverino": "Min. Paulo de Tarso Sanseverino",
    "raul araujo": "Min. Raul Araújo",
    "raul araujo filho": "Min. Raul Araújo",
    "reynaldo soares da fonseca": "Min. Reynaldo Soares da Fonseca",
    "ricardo lewandowski": "Min. Ricardo Lewandowski",
    "rosa weber": "Min. Rosa Weber",
    "vera lucia": "Min. Vera Lúcia Santana Araújo",
    "vera lucia santana araujo": "Min. Vera Lúcia Santana Araújo",
    "edilene lobo": "Min. Edilene Lôbo",
    "ricardo villas boas cueva": "Min. Ricardo Villas Bôas Cueva",
    "villas boas cueva": "Min. Ricardo Villas Bôas Cueva",
    "vilas boas cueva": "Min. Ricardo Villas Bôas Cueva",
    "sergio banhos": "Min. Sérgio Banhos",
    "sergio silveira banhos": "Min. Sérgio Banhos",
    "sebastiao reis junior": "Min. Sebastião Reis Júnior",
    "tarcisio vieira": "Min. Tarcísio Vieira de Carvalho Neto",
    "tarcisio vieira de carvalho": "Min. Tarcísio Vieira de Carvalho Neto",
    "tarcisio vieira de carvalho neto": "Min. Tarcísio Vieira de Carvalho Neto",
    "teori zavascki": "Min. Teori Zavascki",
}
MINISTROS_STF = {
    "Min. Alexandre de Moraes",
    "Min. André Mendonça",
    "Min. Cármen Lúcia",
    "Min. Cristiano Zanin",
    "Min. Dias Toffoli",
    "Min. Edson Fachin",
    "Min. Flávio Dino",
    "Min. Gilmar Mendes",
    "Min. Luiz Fux",
    "Min. Luís Roberto Barroso",
    "Min. Marco Aurélio",
    "Min. Nunes Marques",
    "Min. Ricardo Lewandowski",
    "Min. Rosa Weber",
    "Min. Teori Zavascki",
}
MINISTROS_STJ = {
    "Min. Antônio Carlos Ferreira",
    "Min. Benedito Gonçalves",
    "Min. Herman Benjamin",
    "Min. Humberto Martins",
    "Min. Isabel Gallotti",
    "Min. João Otávio de Noronha",
    "Min. Jorge Mussi",
    "Min. Laurita Vaz",
    "Min. Luís Felipe Salomão",
    "Min. Maria Thereza de Assis Moura",
    "Min. Mauro Campbell Marques",
    "Min. Napoleão Nunes Maia Filho",
    "Min. Og Fernandes",
    "Min. Paulo de Tarso Sanseverino",
    "Min. Raul Araújo",
    "Min. Reynaldo Soares da Fonseca",
    "Min. Ricardo Villas Bôas Cueva",
    "Min. Sebastião Reis Júnior",
}
MINISTROS_JURISTAS = {
    "Min. Admar Gonzaga",
    "Min. Arnaldo Versiani",
    "Min. Carlos Horbach",
    "Min. Carlos Mário da Silva Velloso Filho",
    "Min. Edilene Lôbo",
    "Min. Estela Aranha",
    "Min. Floriano de Azevedo Marques",
    "Min. Henrique Neves da Silva",
    "Min. Humberto Jacques de Medeiros",
    "Min. Joelson Dias",
    "Min. Luciana Lóssio",
    "Min. Maria Cláudia Bucchianeri",
    "Min. Ramos Tavares",
    "Min. Sérgio Banhos",
    "Min. Tarcísio Vieira de Carvalho Neto",
    "Min. Vera Lúcia Santana Araújo",
}
MINISTRO_INVALID_NAME_TERMS = {
    "acompanhado",
    "aplicacao",
    "apresentou",
    "apresenta",
    "aprovacao",
    "aprovou",
    "arguiu",
    "caso",
    "colegiado",
    "conclusao",
    "concluiu",
    "desconhecido",
    "encaminhamento",
    "entendeu",
    "foi",
    "julgamento",
    "julgou",
    "lista",
    "manteve",
    "ministro",
    "ministra",
    "negou",
    "pedido",
    "poder",
    "presidente",
    "processo",
    "provimento",
    "relator",
    "relatora",
    "recurso",
    "sobre",
    "submeteu",
    "vista",
    "voto",
}
MINISTRO_NAME_PARTICLES = {"de", "da", "do", "dos", "das", "e"}

EMPTY_ADVOGADOS_REGEX = re.compile(
    r"(?i)\b("
    r"n[ãa]o\s+(?:citad\w*|mencionad\w*|informad\w*|consta\w*|houve|h[áa]|aplic[aá]vel)"
    r"|sem\s+sustenta[cç][aã]o"
    r"|sem\s+advogad\w*"
    r"|n/?a"
    r")\b"
)
MPE_REFERENCE_REGEX = re.compile(r"(?i)\bminist[ée]rio\s+p[úu]blico\s+eleitoral\b")
MINISTERIO_PUBLICO_REGEX = re.compile(r"(?i)\bminist[ée]rio\s+p[úu]blico\b")
MPE_ABBREV_REGEX = re.compile(r"(?i)\bmp\s*eleitoral\b|\bm\.?\s*p\.?\s*e\.?\b|\bmpe\b")
MPE_REPRESENTATIVE_REGEX = re.compile(
    r"(?i)\b(vice-?procurador(?:-geral)?\s+eleitoral|"
    r"procurador(?:-geral)?(?:\s+regional)?\s+eleitoral|"
    r"procuradoria\s+geral\s+eleitoral|pge)\b"
)
PARTY_ATTORNEY_MARKER_REGEX = re.compile(
    r"(?i)\b("
    r"dr\.?|dra\.?|doutor(?:a)?|adv\.?|advogad[oa]s?|patrono(?:s)?|patrona(?:s)?|"
    r"defensor(?:a|es|as)?|procurador(?:a|es|as)?"
    r")\b"
)
PARTY_OAB_REGEX = re.compile(
    r"(?i)\bOAB(?:/[A-Z]{2})?\s*[-:]?\s*[\d./-]+\b"
)
PARTY_PLACEHOLDER_REGEX = re.compile(
    r"(?i)^\s*("
    r"n[ãa]o\s+especificad\w*|"
    r"n[ãa]o\s+informad\w*|"
    r"n[ãa]o\s+identificad\w*|"
    r"n[ãa]o\s+mencionad\w*|"
    r"n[ãa]o\s+consta\w*|"
    r"n[ãa]o\s+declarad\w*|"
    r"n[ãa]o\s+aplic[aá]ve\w*|"
    r"n[ãa]o\s+h[áa]\b.*|"
    r"desconhecid\w*|"
    r"ignorado\w*|"
    r"sem\s+identifica[cç][aã]o|"
    r"sem\s+informa[cç][aã]o|"
    r"n/?a"
    r")\s*$"
)
PARTY_PROCESSUAL_ROLE_MAP = {
    "agravante": "Agravante",
    "agravantes": "Agravante",
    "agravado": "Agravado",
    "agravada": "Agravada",
    "agravados": "Agravado",
    "agravadas": "Agravada",
    "apelante": "Apelante",
    "apelantes": "Apelante",
    "apelado": "Apelado",
    "apelada": "Apelada",
    "apelados": "Apelado",
    "apeladas": "Apelada",
    "autor": "Autor",
    "autora": "Autora",
    "autores": "Autor",
    "autoras": "Autora",
    "candidato": "Candidato",
    "candidata": "Candidata",
    "candidatos": "Candidato",
    "candidatas": "Candidata",
    "embargante": "Embargante",
    "embargantes": "Embargante",
    "embargado": "Embargado",
    "embargada": "Embargada",
    "embargados": "Embargado",
    "embargadas": "Embargada",
    "impugnante": "Impugnante",
    "impugnantes": "Impugnante",
    "impugnado": "Impugnado",
    "impugnada": "Impugnada",
    "impugnados": "Impugnado",
    "impugnadas": "Impugnada",
    "impetrante": "Impetrante",
    "impetrantes": "Impetrante",
    "impetrado": "Impetrado",
    "impetrada": "Impetrada",
    "impetrados": "Impetrado",
    "impetradas": "Impetrada",
    "interessado": "Interessado",
    "interessada": "Interessada",
    "interessados": "Interessado",
    "interessadas": "Interessada",
    "investigado": "Investigado",
    "investigada": "Investigada",
    "investigados": "Investigado",
    "investigadas": "Investigada",
    "investigante": "Investigante",
    "investigantes": "Investigante",
    "recorrente": "Recorrente",
    "recorrentes": "Recorrente",
    "recorrido": "Recorrido",
    "recorrida": "Recorrida",
    "recorridos": "Recorrido",
    "recorridas": "Recorrida",
    "representante": "Representante",
    "representantes": "Representante",
    "representado": "Representado",
    "representada": "Representada",
    "representados": "Representado",
    "representadas": "Representada",
    "requerente": "Requerente",
    "requerentes": "Requerente",
    "requerido": "Requerido",
    "requerida": "Requerida",
    "requeridos": "Requerido",
    "requeridas": "Requerida",
    "consulente": "Consulente",
    "consulentes": "Consulente",
    "paciente": "Paciente",
    "pacientes": "Paciente",
    "reu": "Réu",
    "réu": "Réu",
    "reus": "Réu",
    "réus": "Réu",
    "re": "Ré",
    "ré": "Ré",
    "res": "Ré",
    "rés": "Ré",
}
PARTY_PROCESSUAL_ROLE_REGEX = re.compile(
    r"(?i)^\s*(?P<label>"
    + "|".join(re.escape(label) for label in PARTY_PROCESSUAL_ROLE_MAP)
    + r")\s*[:\-–]\s*(?P<name>.+?)\s*$"
)
PROCURADOR_GENERIC_REGEX = re.compile(r"(?i)\b(vice-?procurador(?:-geral)?|procurador(?:-geral)?|procuradoria)\b")

FEMALE_NAME_HINTS = {
    "ana", "maria", "mariana", "marina", "carla", "claudia", "clara", "camila",
    "carolina", "caroline", "beatriz", "bianca", "renata", "fernanda", "patricia",
    "luciana", "lucia", "marcia", "sandra", "silvana", "viviane", "oneida",
    "andreia", "adriana", "tatiana", "vanessa", "aline", "leticia", "raquel",
    "cristina", "cristiane", "gisele", "giovana", "gabriela", "isabel", "marta",
    "carmen", "estela", "juliana", "julia", "amanda", "bruna", "daniela",
}
# Nomes MASCULINOS terminados em 'a' (excecoes a heuristica '-a -> feminino' abaixo).
MALE_NAME_HINTS = {
    "juca", "jonata", "nicola", "luca", "joshua", "agripa", "noa", "elia",
    "neemia", "ananias", "josua", "andrea", "cosma", "barnaba",
}
# SOBRENOMES comuns terminados em 'a' — quando aparecem como 1o token (nome incompleto/
# sobrenome-primeiro), NAO inferir feminino por engano (ex.: "Dr. Silva" nao vira "Dra.").
SURNAME_A_ENDING = {
    "silva", "costa", "ferreira", "oliveira", "pereira", "lima", "rocha", "cunha",
    "mota", "motta", "barbosa", "sousa", "souza", "moreira", "teixeira", "vieira",
    "nogueira", "bezerra", "saraiva", "paiva", "fonseca", "franca", "serra", "almeida",
    "miranda", "holanda", "lacerda", "arruda", "rezende", "resende", "guerra", "braga",
    "fraga", "veiga", "aranha", "mendonca", "padilha", "uchoa", "gouveia", "batista",
    "sena", "espindola", "fontoura", "siqueira", "caldeira", "pedrosa", "feitosa",
    "macena", "messa", "lousada", "boanerges", "bda",
}


def normalize_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def clean_label_value(value: str) -> str:
    value = value.replace("**", "").strip()
    value = re.sub(r"^\*+\s*", "", value).strip()
    value = re.sub(r"\s*\*+$", "", value).strip()
    return value.rstrip(".").strip()


def normalize_token(value: str) -> str:
    value = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def normalize_class_text(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value)
    value = normalize_token(value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def extract_chunk_judgment_process_values(judgment: Any) -> list[str]:
    if not isinstance(judgment, dict):
        return []
    values: list[str] = []
    for field_name in ("processo", "processos"):
        raw_value = judgment.get(field_name)
        raw_values: list[Any]
        if isinstance(raw_value, str):
            raw_values = split_csv_like_text(raw_value) or [raw_value]
        elif isinstance(raw_value, Iterable) and not isinstance(raw_value, (str, bytes, dict)):
            raw_values = list(raw_value)
        elif raw_value in {None, ""}:
            raw_values = []
        else:
            raw_values = [raw_value]
        for candidate in raw_values:
            text = str(candidate or "").strip()
            if text:
                values.append(text)
    return dedupe_preserve_order(values)


def get_canonization_data() -> dict[str, Any]:
    global CANON_DATA
    if CANON_DATA is not None:
        return CANON_DATA

    data: dict[str, Any] = {
        "classes": set(),
        "results": set(),
        "class_results": {},
        "class_norm_map": {},
        "result_norm_map": {},
    }
    path = Path(__file__).resolve().parent / CANON_CSV_FILENAME
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            class_results: dict[str, set[str]] = {}
            for row in reader:
                cls = (row.get("classe_processo") or "").strip()
                res = (row.get("resultado") or "").strip()
                if cls:
                    data["classes"].add(cls)
                    class_results.setdefault(cls, set())
                    if res:
                        class_results[cls].add(res)
                if res:
                    data["results"].add(res)
            data["class_results"] = class_results

    data["class_norm_map"] = {
        normalize_class_text(cls): cls for cls in data["classes"]
    }
    data["result_norm_map"] = {
        normalize_class_text(res): res for res in data["results"]
    }
    CANON_DATA = data
    return data


def is_mpe_noise_entry(text: str) -> bool:
    if not text:
        return False
    return bool(
        MPE_REFERENCE_REGEX.search(text)
        or MPE_ABBREV_REGEX.search(text)
        or MPE_REPRESENTATIVE_REGEX.search(text)
        or MINISTERIO_PUBLICO_REGEX.search(text)
        or PROCURADOR_GENERIC_REGEX.search(text)
    )


def normalize_mpe_reference(value: str) -> str:
    if not value:
        return ""
    value = re.sub(MPE_REFERENCE_REGEX, "MPE", value)
    value = re.sub(MPE_ABBREV_REGEX, "MPE", value)
    return value


def remove_mpe_from_partes(value: str) -> str:
    if not value:
        return ""
    normalized = normalize_mpe_reference(value)
    parts = [part.strip() for part in re.split(r"\s*,\s*|\s*;\s*", normalized) if part.strip()]
    cleaned: list[str] = []
    for part in parts:
        if part == "MPE" or is_mpe_noise_entry(part):
            continue
        cleaned.append(part)
    return ", ".join(dedupe_preserve_order(cleaned))


def looks_like_advogado_party_entry(value: str) -> bool:
    normalized = normalize_mpe_reference(value or "")
    if not normalized:
        return False
    if PARTY_OAB_REGEX.search(normalized):
        return True
    if PARTY_ATTORNEY_MARKER_REGEX.search(normalized):
        return True
    return bool(re.search(r"(?i)\badvogad[oa]\b", normalized))


def _cleanup_party_name(value: str) -> str:
    cleaned = PARTY_OAB_REGEX.sub("", value or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;,:-")
    return cleaned.strip()


def _normalize_party_role_label(value: str) -> str:
    normalized = normalize_token(value or "")
    return PARTY_PROCESSUAL_ROLE_MAP.get(normalized, value.strip())


_ROLE_TITLE_TOKENS = {
    "candidato", "candidata", "candidatos", "candidatas",
    "prefeito", "prefeita", "prefeitos", "prefeitas",
    "vice", "vereador", "vereadora", "vereadores", "vereadoras",
    "deputado", "deputada", "deputados", "deputadas", "federal", "estadual", "distrital",
    "governador", "governadora", "senador", "senadora",
    "conselheiro", "conselheira", "suplente", "suplentes", "presidente", "presidenta", "chefe",
}
_ROLE_LEADING_TOKENS = _ROLE_TITLE_TOKENS | {"e", "a", "ao", "o", "os", "as", "cargo"}
_PLACE_LEADING_TOKENS = {"de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas"}


def is_descriptive_role_noise(value: str) -> bool:
    """True quando o valor é uma FRASE DESCRITIVA de cargo/candidatura, sem nome de
    pessoa (ex.: 'Candidato ao cargo de Deputado...', 'Prefeito e Vice-Prefeito de
    Baixo Guandu/ES', 'Deputado Federal e Vereador'). NÃO considera ruído nomes
    próprios com cargo ('Prefeito João Silva') nem entidades ('Município de ...')."""
    norm = normalize_class_text(value)
    if not norm:
        return False
    if re.search(r"\b(ao cargo de|nas eleicoes|na eleicao de)\b", norm):
        return True
    # Referência a instituição (tribunal, TRE, OAB, ministério, federação...) é parte
    # legítima — ex.: 'Presidente do TRE-RR (autoridade coatora)'. Não expurga.
    if re.search(r"\b(tribunal|tre|tse|stf|stj|oab|incra|minist[eé]rio|procuradoria|conselho|assembleia|federa[cç])\b", norm):
        return False
    tokens = norm.split()
    index = 0
    started_with_role = False
    while index < len(tokens) and tokens[index] in _ROLE_LEADING_TOKENS:
        if tokens[index] in _ROLE_TITLE_TOKENS:
            started_with_role = True
        index += 1
    if not started_with_role:
        return False
    remaining = tokens[index:]
    if not remaining:
        return True  # só cargos/conectores (ex.: 'Deputado Federal e Vereador')
    # cargo seguido de "de <lugar>" (ex.: 'Prefeito ... de Baixo Guandu') é descrição.
    return remaining[0] in _PLACE_LEADING_TOKENS


def standardize_tribunal_party_name(value: str) -> str:
    """Padroniza referências a tribunal eleitoral como PARTE: 'Tribunal Regional
    Eleitoral de Sergipe (TRE-SE)' -> 'TRE/SE'; 'Tribunal Superior Eleitoral' ->
    'TSE'. Retorna '' quando não é exatamente o tribunal (ex.: 'Corregedoria...',
    'Presidente do TRE...' permanecem como estão)."""
    raw = normalize_text(str(value or "")).strip()
    if not raw:
        return ""
    low = normalize_class_text(raw)
    if low.startswith("tribunal superior eleitoral") or re.fullmatch(r"tse(\s*\(tse\))?", low):
        return "TSE"
    if low.startswith("tribunal regional eleitoral"):
        sigla = re.search(r"\btre[\s/_-]*([a-z]{2})\b", low)
        if sigla:
            return f"TRE/{sigla.group(1).upper()}"
        state_match = re.search(r"tribunal regional eleitoral d[eoa]s?\s+(.+?)(?:\s*\(|$)", low)
        if state_match:
            state = state_match.group(1).strip()
            # Match EXATO do nome do estado (mais longos primeiro) para não confundir
            # "Pará" (PA) com "Paraíba" (PB) via substring.
            for name, uf in sorted(STATE_UF.items(), key=lambda kv: -len(kv[0])):
                if state == name or state.startswith(name + " "):
                    return f"TRE/{uf}"
    return ""


def normalize_party_entry(value: str) -> str:
    if not value:
        return ""
    cleaned = normalize_mpe_reference(value)
    cleaned = re.sub(r"(?i)^\s*part[ea]\s*[:\-]\s*", "", cleaned).strip()
    if not cleaned or cleaned == "MPE" or is_mpe_noise_entry(cleaned) or PARTY_PLACEHOLDER_REGEX.match(cleaned):
        return ""
    if is_descriptive_role_noise(cleaned):
        return ""
    tribunal = standardize_tribunal_party_name(cleaned)
    if tribunal:
        return tribunal
    match = PARTY_PROCESSUAL_ROLE_REGEX.match(cleaned)
    if match:
        role_label = _normalize_party_role_label(match.group("label"))
        name = _cleanup_party_name(match.group("name"))
        if not name or is_mpe_noise_entry(name) or looks_like_advogado_party_entry(name) or PARTY_PLACEHOLDER_REGEX.match(name):
            return ""
        return f"{name} ({role_label})"
    suffix_match = re.match(r"^(?P<name>.+?)\s*\((?P<label>[^()]+)\)\s*$", cleaned)
    if suffix_match and normalize_token(suffix_match.group("label")) in PARTY_PROCESSUAL_ROLE_MAP:
        role_label = _normalize_party_role_label(suffix_match.group("label"))
        name = _cleanup_party_name(suffix_match.group("name"))
        if not name or is_mpe_noise_entry(name) or looks_like_advogado_party_entry(name) or PARTY_PLACEHOLDER_REGEX.match(name):
            return ""
        return f"{name} ({role_label})"
    cleaned = _cleanup_party_name(cleaned)
    if not cleaned or looks_like_advogado_party_entry(cleaned) or is_mpe_noise_entry(cleaned) or PARTY_PLACEHOLDER_REGEX.match(cleaned):
        return ""
    return cleaned


PARTY_SPLIT_ORGANIZATION_REGEX = re.compile(
    r"(?i)\b(coliga(?:ção|cao)?|partido|federa(?:ção|cao)?|frente|minist[eé]rio|prefeitura|c[aâ]mara|munic[ií]pio|tribunal|uni[aã]o|governo)\b"
)
PARTY_NAME_CONNECTORS = {"e", "de", "da", "do", "das", "dos"}
TRAILING_PAREN_GROUP_REGEX = re.compile(r"\s*\(([^()]*)\)\s*$")
PARTY_DESCRIPTIVE_PARENTHETICAL_TERMS = {
    "candidato",
    "candidata",
    "eleito",
    "eleita",
    "prefeito",
    "prefeita",
    "vice prefeito",
    "vice prefeita",
    "governador",
    "governadora",
    "senador",
    "senadora",
    "vereador",
    "vereadora",
    "deputado",
    "deputada",
}


def _split_top_level_text(text: str, *, delimiters: str = "", split_conjunction: bool = False) -> list[str]:
    parts: list[str] = []
    buffer = ""
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if depth == 0 and split_conjunction and text[index:index + 3].lower() == " e ":
            if buffer.strip():
                parts.append(buffer.strip())
            buffer = ""
            index += 3
            continue
        if depth == 0 and delimiters and char in delimiters:
            if buffer.strip():
                parts.append(buffer.strip())
            buffer = ""
            index += 1
            continue
        buffer += char
        index += 1
    if buffer.strip():
        parts.append(buffer.strip())
    return parts


def _strip_trailing_parenthetical_groups(text: str) -> tuple[str, list[str]]:
    remaining = text.strip()
    groups: list[str] = []
    while True:
        match = TRAILING_PAREN_GROUP_REGEX.search(remaining)
        if not match:
            break
        groups.insert(0, match.group(1).strip())
        remaining = remaining[:match.start()].rstrip()
    return remaining, groups


def _rebuild_with_parenthetical_groups(base: str, groups: list[str]) -> str:
    result = base.strip()
    for group in groups:
        if group:
            result = f"{result} ({group})" if result else f"({group})"
    return result.strip()


def _is_party_entity_parenthetical_group(value: str) -> bool:
    label = str(value or "").strip()
    if not label:
        return False
    normalized = normalize_token(label)
    if normalized in PARTY_PROCESSUAL_ROLE_MAP:
        return False
    if any(term in normalized for term in PARTY_DESCRIPTIVE_PARENTHETICAL_TERMS):
        return False
    if re.fullmatch(r"[A-Z0-9][A-Z0-9./&+ -]{1,40}", label):
        return True
    return bool(re.search(r"\b[A-Z]{2,}\b$", label))


def canonicalize_party_option_label(value: str) -> str:
    cleaned = normalize_mpe_reference(value or "")
    cleaned = normalize_text(cleaned).strip().strip(" .;,:-")
    if not cleaned or cleaned == "MPE" or is_mpe_noise_entry(cleaned) or PARTY_PLACEHOLDER_REGEX.match(cleaned):
        return ""
    cleaned = re.sub(r"(?i)^\s*part[ea]\s*[:\-]\s*", "", cleaned).strip()
    prefix_match = PARTY_PROCESSUAL_ROLE_REGEX.match(cleaned)
    if prefix_match:
        cleaned = prefix_match.group("name").strip()
    cleaned = re.sub(r"(?i)^\s*coliga(?:ção|cao)\s*[:\-]\s*", "Coligação ", cleaned).strip()
    cleaned = re.sub(r"(?i)\b(coliga(?:ção|cao)\s+)['‘’]([^'‘’]+)['‘’]", r"\1\2", cleaned)
    cleaned = re.sub(r"[\"“”]", "", cleaned)
    if cleaned.count("(") < cleaned.count(")"):
        cleaned = re.sub(r"\s*-\s*([A-Z0-9]{2,}(?:/[A-Z0-9]{2,})*)\)$", r" (\1)", cleaned)
    base, groups = _strip_trailing_parenthetical_groups(cleaned)
    kept_groups: list[str] = []
    for group in groups:
        normalized_group = normalize_token(group)
        if normalized_group in PARTY_PROCESSUAL_ROLE_MAP:
            continue
        if any(term in normalized_group for term in PARTY_DESCRIPTIVE_PARENTHETICAL_TERMS):
            continue
        if _is_party_entity_parenthetical_group(group):
            kept_groups.append(group)
            continue
        kept_groups.append(group)
    cleaned = _rebuild_with_parenthetical_groups(base, kept_groups)
    if cleaned.count("(") < cleaned.count(")"):
        cleaned = re.sub(r"\s*-\s*([A-Z0-9]{2,}(?:/[A-Z0-9]{2,})*)\)$", r" (\1)", cleaned)
    if cleaned.count("(") > cleaned.count(")"):
        tail = cleaned.rsplit("(", 1)[-1].strip()
        if re.fullmatch(r"[A-Z0-9./&+ -]{2,40}", tail) or re.search(r"\b[A-Z0-9]{2,}$", tail):
            cleaned = f"{cleaned})"
    cleaned = re.sub(r"(?i)\s+e\s+demais$", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\s+e\s+outr[oa]s?$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;,:-")
    if not cleaned or cleaned == "MPE" or is_mpe_noise_entry(cleaned) or PARTY_PLACEHOLDER_REGEX.match(cleaned):
        return ""
    if looks_like_advogado_party_entry(cleaned):
        return ""
    return cleaned


def _looks_like_person_name_segment(value: str) -> bool:
    candidate = normalize_text(str(value or "")).strip()
    if not candidate or PARTY_SPLIT_ORGANIZATION_REGEX.search(candidate):
        return False
    candidate, _groups = _strip_trailing_parenthetical_groups(candidate)
    candidate = re.sub(r"(?i)^(dr\.?|dra\.?|doutor(?:a)?|advogad[oa]s?)\s+", "", candidate).strip()
    words = [word.strip(".,;:") for word in candidate.split() if word.strip(".,;:")]
    if len(words) < 2:
        return False
    non_connectors = [word for word in words if normalize_token(word) not in PARTY_NAME_CONNECTORS]
    if len(non_connectors) < 2:
        return False
    return all(
        normalize_token(word) in PARTY_NAME_CONNECTORS or bool(re.match(r"^[A-ZÀ-Ý]", word))
        for word in words
    )


def _extract_party_role_suffix(value: str) -> tuple[str, str]:
    base, groups = _strip_trailing_parenthetical_groups(value)
    for index in range(len(groups) - 1, -1, -1):
        label = groups[index]
        if normalize_token(label) in PARTY_PROCESSUAL_ROLE_MAP:
            role = _normalize_party_role_label(label)
            remaining_groups = [group for group_index, group in enumerate(groups) if group_index != index]
            return _rebuild_with_parenthetical_groups(base, remaining_groups), role
    return value.strip(), ""


def _extract_advogado_shared_suffix(value: str) -> tuple[str, str]:
    base, groups = _strip_trailing_parenthetical_groups(value)
    for index in range(len(groups) - 1, -1, -1):
        normalized_label = _normalize_advogado_label_hint(groups[index])
        if normalized_label:
            remaining_groups = [group for group_index, group in enumerate(groups) if group_index != index]
            return _rebuild_with_parenthetical_groups(base, remaining_groups), normalized_label
    return value.strip(), ""


def split_conjoined_person_party_entry(value: str) -> list[str]:
    text = normalize_text(str(value or "")).strip()
    if not text:
        return []
    if PARTY_SPLIT_ORGANIZATION_REGEX.search(text):
        return [text]
    parts = [part.strip(" ,;") for part in _split_top_level_text(text, split_conjunction=True) if part.strip(" ,;")]
    if len(parts) < 2:
        return [text]
    split_parts = [list(_extract_party_role_suffix(part)) for part in parts]
    distinct_roles = {role for _base, role in split_parts if role}
    shared_role = next(iter(distinct_roles)) if len(distinct_roles) == 1 and sum(1 for _base, role in split_parts if role) == 1 else ""
    normalized_parts: list[str] = []
    for base, own_role in split_parts:
        candidate = base.strip()
        if not _looks_like_person_name_segment(candidate):
            return [text]
        effective_role = own_role or shared_role
        if effective_role:
            candidate = f"{candidate} ({effective_role})"
        normalized_parts.append(candidate)
    if normalized_parts:
        return normalized_parts
    return [text]


def _flatten_structured_party_payload(value: Any, label_hint: str = "") -> list[str]:
    if isinstance(value, dict):
        flattened: list[str] = []
        for key, nested in value.items():
            key_text = str(key or "").strip().strip("'\"")
            next_label = "" if normalize_token(key_text) in {"parte", "partes"} else key_text
            flattened.extend(_flatten_structured_party_payload(nested, next_label or label_hint))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_structured_party_payload(item, label_hint))
        return flattened
    text = str(value or "").strip().strip("'\"")
    if not text:
        return []
    if label_hint and not PARTY_PROCESSUAL_ROLE_REGEX.match(text):
        return [f"{label_hint}: {text}"]
    return [text]


def _looks_like_structured_party_fragment(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith(("{", "[")):
        return True
    return bool(re.match(r"""^\s*['"][^'"]+['"]\s*:\s*""", text))


def _parse_structured_partes_payload(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if not text.startswith(("{", "[")):
        label_match = re.match(r"""^\s*['"]?(?P<label>[^'":{}\[\],]+)['"]?\s*:\s*(?P<raw>.+?)\s*$""", text)
        if not label_match:
            return []
        label = label_match.group("label").strip()
        label_norm = normalize_token(label)
        if label_norm not in PARTY_PROCESSUAL_ROLE_MAP:
            return []
        raw_value = label_match.group("raw").strip().rstrip("}").strip()
        if not raw_value:
            return []
        parsed_value: Any
        try:
            parsed_value = ast.literal_eval(raw_value)
        except Exception:
            cleaned = raw_value.strip().strip("'\"")
            if not cleaned:
                return []
            values = [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip().strip("'\"")]
            parsed_value = values if len(values) > 1 else cleaned
        else:
            if isinstance(parsed_value, str) and "," in parsed_value and label_norm.endswith("s"):
                values = [item.strip().strip("'\"") for item in parsed_value.split(",") if item.strip().strip("'\"")]
                if len(values) > 1:
                    parsed_value = values
        return _flatten_structured_party_payload({label: parsed_value})
    try:
        payload = ast.literal_eval(text)
    except Exception:
        return []
    return _flatten_structured_party_payload(payload)


def normalize_partes_list(value: str | list[str]) -> str:
    if isinstance(value, list):
        raw_parts = [str(item or "").strip() for item in value if str(item or "").strip()]
        structured_entries = []
        if raw_parts and any(_looks_like_structured_party_fragment(part) for part in raw_parts):
            structured_entries = _parse_structured_partes_payload(", ".join(raw_parts))
        if structured_entries:
            raw_parts = structured_entries
    else:
        structured_entries = _parse_structured_partes_payload(str(value or ""))
        if structured_entries:
            raw_parts = structured_entries
        else:
            raw_parts = [part.strip() for part in re.split(r"\s*,\s*|\s*;\s*", str(value or "")) if part.strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        entry = normalize_party_entry(part)
        if not entry:
            continue
        for split_entry in split_conjoined_person_party_entry(entry):
            if not split_entry or split_entry in seen:
                continue
            seen.add(split_entry)
            normalized.append(split_entry)
    return ", ".join(normalized)


def is_empty_advogados_value(value: str) -> bool:
    if not value:
        return True
    return bool(EMPTY_ADVOGADOS_REGEX.search(value))


def infer_advogado_prefix(name: str, label_hint: str = "") -> str:
    hint = label_hint.lower()
    if "advogada" in hint:
        return "Dra."
    if "advogado" in hint:
        return "Dr."
    first = normalize_token(name.split()[0]) if name.split() else ""
    if first in FEMALE_NAME_HINTS:
        return "Dra."
    if first in MALE_NAME_HINTS or first in SURNAME_A_ENDING:
        return "Dr."
    # Heuristica do portugues: 1o nome terminado em 'a' (>=3 letras, fora das excecoes
    # masculinas e de sobrenomes) costuma ser feminino -> Dra. Cobre Sara/Laura/Nadja/etc.
    if len(first) >= 3 and first.endswith("a"):
        return "Dra."
    return "Dr."


def _looks_like_advogado_label(value: str) -> bool:
    cleaned = normalize_text(str(value or "")).strip().strip("'\"")
    if not cleaned:
        return False
    normalized = normalize_token(cleaned.replace("-", " ").replace("_", " "))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in PARTY_PROCESSUAL_ROLE_MAP:
        return True
    return normalized.startswith((
        "pelo ",
        "pela ",
        "advogado do ",
        "advogada do ",
        "advogado da ",
        "advogada da ",
        "advogado dos ",
        "advogada dos ",
        "advogado das ",
        "advogada das ",
        "representante do ",
        "representante da ",
        "representante dos ",
        "representante das ",
    ))


def _normalize_advogado_label_hint(value: str) -> str:
    cleaned = normalize_text(str(value or "")).strip().strip("'\"")
    if not cleaned:
        return ""
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    normalized = normalize_token(cleaned)
    if normalized in PARTY_PROCESSUAL_ROLE_MAP:
        role = PARTY_PROCESSUAL_ROLE_MAP[normalized].lower()
        article = "pela" if role.endswith("a") or role == "ré" else "pelo"
        return f"{article} {role}"
    if _looks_like_advogado_label(cleaned):
        return cleaned
    return ""


def _cleanup_structured_advogado_text(value: str) -> str:
    text = normalize_text(str(value or "")).strip()
    if not text:
        return ""
    if "{" in text and not text.lstrip().startswith(("{", "[")):
        text = text[text.index("{"):]
    text = re.sub(r"(?i)^\s*(dr\.?|dra\.?)\s+(?=[{\[])", "", text)
    text = re.sub(r"(?i)([{,]\s*)(dr\.?|dra\.?)\s+(?=['\"])", r"\1", text)
    return text.strip().rstrip(",")


def _flatten_structured_advogado_payload(value: Any, label_hint: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        flattened: list[tuple[str, str]] = []
        for key, nested in value.items():
            key_text = str(key or "").strip().strip("'\"")
            next_label = "" if normalize_token(key_text) in {"advogado", "advogados"} else key_text
            flattened.extend(_flatten_structured_advogado_payload(nested, next_label or label_hint))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened: list[tuple[str, str]] = []
        for item in value:
            flattened.extend(_flatten_structured_advogado_payload(item, label_hint))
        return flattened
    text = str(value or "").strip().strip("'\"")
    if not text:
        return []
    return [(text, label_hint)]


def _looks_like_structured_advogado_fragment(value: str) -> bool:
    text = normalize_text(str(value or "")).strip()
    if not text:
        return False
    if text.startswith(("{", "[")) or "{" in text:
        return True
    match = re.match(r"""^\s*['"]?(?P<label>[^'":{}\[\],]+)['"]?\s*:\s*""", text)
    return bool(match and _looks_like_advogado_label(match.group("label")))


def _parse_structured_advogados_payload(value: str) -> list[tuple[str, str]]:
    text = _cleanup_structured_advogado_text(value)
    if not text:
        return []
    if not text.startswith(("{", "[")):
        label_match = re.match(r"""^\s*['"]?(?P<label>[^'":{}\[\],]+)['"]?\s*:\s*(?P<raw>.+?)\s*$""", text)
        if not label_match:
            return []
        label = label_match.group("label").strip()
        if not _looks_like_advogado_label(label):
            return []
        raw_value = label_match.group("raw").strip().rstrip("}").strip()
        if not raw_value:
            return []
        parsed_value: Any
        try:
            parsed_value = ast.literal_eval(raw_value)
        except Exception:
            cleaned = raw_value.strip().strip("'\"")
            if not cleaned:
                return []
            values = [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip().strip("'\"")]
            parsed_value = values if len(values) > 1 else cleaned
        return _flatten_structured_advogado_payload({label: parsed_value})
    try:
        payload = ast.literal_eval(text)
    except Exception:
        return []
    return _flatten_structured_advogado_payload(payload)


def _extract_advogado_label_and_name(value: str) -> tuple[str, str]:
    text = normalize_text(str(value or "")).strip()
    match = re.match(r"""^\s*['"]?(?P<label>[^'":{}\[\],]+)['"]?\s*:\s*(?P<name>.+?)\s*$""", text)
    if not match:
        return "", text
    label = match.group("label").strip()
    if not _looks_like_advogado_label(label):
        return "", text
    return label, match.group("name").strip()


def normalize_advogado_name(name: str, label_hint: str = "") -> str:
    name = name.strip()
    if not name or is_mpe_noise_entry(name) or is_empty_advogados_value(name):
        return ""
    suffix = ""
    if "(" in name:
        base, extra = name.split("(", 1)
        name = base.strip()
        suffix = " (" + extra.strip()
    name = name.strip().strip("{}[]")
    name = name.strip().strip("'\"")
    name = name.rstrip(".;:,").strip()
    if not name or is_empty_advogados_value(name):
        return ""
    prefix = ""
    if re.match(r"(?i)^dra\.?\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^dra\.?\s+", "", name).strip()
    elif re.match(r"(?i)^dr\.?\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^dr\.?\s+", "", name).strip()
    elif re.match(r"(?i)^doutora\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^doutora\s+", "", name).strip()
    elif re.match(r"(?i)^doutor\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^doutor\s+", "", name).strip()
    elif re.match(r"(?i)^(sra|srta|senhora)\.?\s+", name):
        prefix = "Dra."
        name = re.sub(r"(?i)^(sra|srta|senhora)\.?\s+", "", name).strip()
    elif re.match(r"(?i)^(sr|senhor)\.?\s+", name):
        prefix = "Dr."
        name = re.sub(r"(?i)^(sr|senhor)\.?\s+", "", name).strip()
    if not name:
        return ""
    if not prefix:
        prefix = infer_advogado_prefix(name, label_hint)
    label_suffix = _normalize_advogado_label_hint(label_hint)
    if label_suffix:
        normalized_suffix = normalize_token(suffix)
        if normalize_token(label_suffix) not in normalized_suffix:
            suffix = f"{suffix} ({label_suffix})" if suffix else f" ({label_suffix})"
    return f"{prefix} {name}{suffix}".strip()


def split_advogados_entries(text: str) -> list[str]:
    normalized_text = normalize_text(str(text or "")).strip()
    if not normalized_text:
        return []
    comma_split_parts = [part.strip() for part in _split_top_level_text(normalized_text, delimiters=",;") if part.strip()]
    parts: list[str] = []
    for raw_part in comma_split_parts:
        if PARTY_SPLIT_ORGANIZATION_REGEX.search(raw_part) and not PARTY_ATTORNEY_MARKER_REGEX.search(raw_part):
            parts.append(raw_part)
            continue
        and_split_parts = [part.strip() for part in _split_top_level_text(raw_part, split_conjunction=True) if part.strip()]
        if len(and_split_parts) >= 2 and all(_looks_like_person_name_segment(part) for part in and_split_parts):
            shared_suffix_parts = [list(_extract_advogado_shared_suffix(part)) for part in and_split_parts]
            distinct_labels = {label for _base, label in shared_suffix_parts if label}
            shared_label = (
                next(iter(distinct_labels))
                if len(distinct_labels) == 1 and sum(1 for _base, label in shared_suffix_parts if label) == 1
                else ""
            )
            for base, own_label in shared_suffix_parts:
                candidate = base.strip()
                effective_label = own_label or shared_label
                if effective_label:
                    candidate = f"{candidate} ({effective_label})"
                parts.append(candidate)
            continue
        parts.append(raw_part)
    return parts


def normalize_advogados_list(value: str | list[str], label_hint: str = "") -> str:
    if not value:
        return ""
    raw_pairs: list[tuple[str, str]] = []
    if isinstance(value, list):
        raw_parts = [normalize_text(str(item or "")).strip() for item in value if normalize_text(str(item or "")).strip()]
        if raw_parts and any(_looks_like_structured_advogado_fragment(part) for part in raw_parts):
            for part in raw_parts:
                raw_pairs.extend(_parse_structured_advogados_payload(part))
            if not raw_pairs:
                raw_pairs = _parse_structured_advogados_payload(", ".join(raw_parts))
        if not raw_pairs:
            raw_pairs = [(part, label_hint) for part in raw_parts]
    else:
        if is_empty_advogados_value(value) and not re.search(r"(?i)[,;]|\bdr\.|\bdra\.|\be\b", value):
            return ""
        structured_pairs = _parse_structured_advogados_payload(str(value or ""))
        if structured_pairs:
            raw_pairs = structured_pairs
        else:
            raw_pairs = [(normalize_text(str(value or "")), label_hint)]
    normalized: list[str] = []
    for raw_text, raw_label in raw_pairs:
        for part in split_advogados_entries(raw_text):
            part = re.sub(r"(?i)^(advogad[oa]s?|defensor[oa]s?)\s*:?\s*", "", part).strip()
            if not part or is_mpe_noise_entry(part) or is_empty_advogados_value(part):
                continue
            entry_label, entry_name = _extract_advogado_label_and_name(part)
            normalized_name = normalize_advogado_name(entry_name, entry_label or raw_label or label_hint)
            if normalized_name:
                normalized.append(normalized_name)
    return ", ".join(dedupe_preserve_order(normalized))


def extract_full_cnj(text: str) -> str:
    match = re.search(CNJ_REGEX, normalize_text(text))
    return match.group(0) if match else ""


def extract_short_processo(text: str) -> str:
    match = re.search(SHORT_PROCESSO_REGEX, normalize_text(text))
    return match.group(0) if match else ""


def extract_labeled_short_processo(text: str) -> str:
    match = re.search(LABELED_PROCESSO_REGEX, normalize_text(text))
    return match.group(1) if match else ""


def extract_special_processo(text: str) -> str:
    match = re.search(SPECIAL_PROCESSO_REGEX, normalize_text(text))
    if not match:
        return ""
    label = match.group("label").upper()
    number = match.group("number").lstrip("0") or "0"
    return f"{label} {number}"


def extract_labeled_short_processo_with_class(text: str) -> str:
    match = re.search(LABELED_SHORT_PROCESSO_REGEX, normalize_text(text))
    if not match:
        return ""
    return format_short_process_number_from_digits(match.group(1))


def format_short_process_number_from_digits(value: str) -> str:
    digits = re.sub(r"\D", "", normalize_text(value))
    if 5 <= len(digits) <= 7:
        return f"{digits[:-2]}-{digits[-2:]}"
    if len(digits) == 8:
        # Some extractions drop the fourth digit of the electoral short number,
        # e.g. 06007196 instead of 060007196. We only repair this pattern for
        # the common 060-prefix family; other 8-digit shorts like 60350714 are
        # left untouched because they are already valid in the dataset.
        prefix = digits[:-2]
        if digits.startswith("060"):
            prefix = f"{prefix[:3]}0{prefix[3:]}"
        return f"{prefix}-{digits[-2:]}"
    if len(digits) == 9:
        return f"{digits[:-2]}-{digits[-2:]}"
    if 10 <= len(digits) <= 12:
        # Some model outputs leak extra leading digits while preserving a valid
        # short process number in the final 9-digit suffix, e.g. 060061316874.
        suffix = digits[-9:]
        if suffix.startswith("0"):
            return f"{suffix[:-2]}-{suffix[-2:]}"
    return ""


def normalize_numero_processo_display(value: str) -> str:
    if not value:
        return ""
    special = extract_special_processo(value)
    if special:
        return special
    full_cnj = extract_full_cnj(value)
    if full_cnj:
        return full_cnj
    short = extract_short_processo(value)
    if short:
        normalized_short = format_short_process_number_from_digits(short)
        return normalized_short if normalized_short else short
    labeled_short = extract_labeled_short_processo_with_class(value)
    if labeled_short:
        return labeled_short
    digits_short = format_short_process_number_from_digits(value)
    if digits_short:
        return digits_short
    labeled = extract_labeled_short_processo(value)
    if labeled and re.search(r"\d", labeled):
        labeled_formatted = format_short_process_number_from_digits(labeled)
        return labeled_formatted if labeled_formatted else labeled
    return ""


def normalize_processo_num(value: str) -> str:
    if not value:
        return ""
    special = extract_special_processo(value)
    if special:
        return special
    full_cnj = extract_full_cnj(value)
    if full_cnj:
        short = extract_short_processo(full_cnj)
        return short if short else full_cnj
    short = extract_short_processo(value)
    if short:
        normalized_short = format_short_process_number_from_digits(short)
        return normalized_short if normalized_short else short
    labeled_short = extract_labeled_short_processo_with_class(value)
    if labeled_short:
        return labeled_short
    digits_short = format_short_process_number_from_digits(value)
    if digits_short:
        return digits_short
    labeled = extract_labeled_short_processo(value)
    if labeled and re.search(r"\d", labeled):
        labeled_formatted = format_short_process_number_from_digits(labeled)
        return labeled_formatted if labeled_formatted else labeled
    return ""


def canonicalize_numero_processo(value: str) -> str:
    return normalize_processo_num(value)


def extract_uf_from_text(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\bUF\s*:\s*([A-Z]{2})\b", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"/([A-Z]{2})\b", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\(([A-Z]{2})\)", text)
    if match:
        return match.group(1).upper()
    match = re.search(
        r"Tribunal Regional Eleitoral d(?:e|o|a)\s+([A-Za-zçÇãõéíóúÁÉÍÓÚ ]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        state_name = match.group(1).strip().lower()
        uf = STATE_UF.get(state_name)
        if uf:
            return uf
    lower = text.lower()
    for state_name in sorted(STATE_UF.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(state_name)}\b", lower):
            return STATE_UF[state_name]
    return ""


def normalize_origem_value(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value).strip().rstrip(".")
    if not value:
        return ""
    comma_parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(comma_parts) >= 3:
        return ""
    normalized_key = normalize_class_text(value)
    if "tribunais regionais eleitorais" in normalized_key:
        return ""
    if normalized_key in {"tribunal superior eleitoral", "tse"}:
        return UF_CAPITALS["DF"]
    if re.fullmatch(r"[A-Za-z]{2}", value) and value.upper() in UF_CAPITALS:
        return UF_CAPITALS[value.upper()]
    if normalized_key in STATE_NAME_KEYS:
        return UF_CAPITALS.get(STATE_UF[normalized_key], "")
    if normalized_key in CAPITAL_NAME_TO_VALUE:
        return CAPITAL_NAME_TO_VALUE[normalized_key]
    zona_match = re.search(r"(?i)(?:\d+\S*\s+)?zona eleitoral de?\s+([^/]+)/([a-z]{2})", value)
    if zona_match:
        city = zona_match.group(1).strip(" ,.;:-")
        uf = zona_match.group(2).upper()
        if city:
            return f"{city}/{uf}"
    prefixed_patterns = [
        r"(?i)^(?:decisoes?|decisões|jurisprudencia|jurisprudência)\s+d[oa]\s+(.+)$",
        r"(?i)^municip(?:al|io)\s+de\s+(.+)$",
        r"(?i)^tribunal de justi[cç]a d(?:e|o|a)\s+(.+)$",
        r"(?i)^tribunal de justi[cç]a do estado d[eo]\s+(.+)$",
        r"(?i)^ju[ií]zo eleitoral d[eo]\s+(.+)$",
        r"(?i)^zona eleitoral d[ea]?\s+(.+)$",
        r"(?i)^prefeitura(?:\s+municipal)?\s+de\s+(.+)$",
        r"(?i)^prefeito(?:\s+e\s+vice-prefeito)?\s+de\s+(.+)$",
    ]
    for pattern in prefixed_patterns:
        match = re.match(pattern, value)
        if not match:
            continue
        stripped = match.group(1).strip(" ,.;:-")
        normalized = normalize_origem_value(stripped)
        if normalized:
            return normalized
        value = stripped
        normalized_key = normalize_class_text(value)
        break
    tse_uf_match = re.match(r"(?i)^tse[-/\s]?([a-z]{2})$", value)
    if tse_uf_match:
        return UF_CAPITALS.get(tse_uf_match.group(1).upper(), "")
    tre_context_match = re.match(
        r"(?i)^(?:titular|suplente|ju[ií]z(?:a)?|decisoes?|decisões|jurisprudencia|jurisprudência)\s+d[oa]\s+tre[-/\s]?([a-z]{2})$",
        value,
    )
    if tre_context_match:
        return UF_CAPITALS.get(tre_context_match.group(1).upper(), "")
    tit_tre_match = re.search(r"(?i)\btre[-/\s]?([a-z]{2})\b", value)
    if tit_tre_match and any(marker in normalized_key for marker in {"titular do tre", "suplente do tre", "juiz do tre"}):
        return UF_CAPITALS.get(tit_tre_match.group(1).upper(), "")
    eleitoral_match = re.match(r"(?i)^eleitoral\s+de\s+(.+)$", value)
    if eleitoral_match:
        value = eleitoral_match.group(1).strip()
        normalized_key = normalize_class_text(value)
        if normalized_key in STATE_NAME_KEYS:
            return UF_CAPITALS.get(STATE_UF[normalized_key], "")
    if re.search(r"(?i)^tribunal regional eleitoral d(?:e|o|a)\s+", value):
        uf = extract_uf_from_text(value)
        return UF_CAPITALS.get(uf, "") if uf else ""
    tre_match = re.match(r"(?i)^tre[-/\s]?([a-z]{2})$", value)
    if tre_match:
        return UF_CAPITALS.get(tre_match.group(1).upper(), "")
    if re.match(r"^.+\([^)]+\)\s*$", value):
        uf = extract_uf_from_text(value)
        city = re.sub(r"\s*\([^)]+\)\s*$", "", value).strip()
        if city and uf:
            return f"{city}/{uf}"
    match = re.match(r"^(.*?)\s*/\s*([A-Za-z]{2})$", value)
    if match:
        return f"{match.group(1).strip()}/{match.group(2).upper()}"
    match = re.match(r"^(.*?)\s*[-–—]\s*([A-Za-z]{2})$", value)
    if match:
        return f"{match.group(1).strip()}/{match.group(2).upper()}"
    match = re.match(r"^(.*?)\s*,\s*([A-Za-z]{2})$", value)
    if match:
        return f"{match.group(1).strip()}/{match.group(2).upper()}"
    match = re.match(r"^(.*?)\s*,\s*([A-Za-zÀ-ÿ ]+)$", value)
    if match:
        city = match.group(1).strip()
        uf = extract_uf_from_text(match.group(2).strip())
        if city and uf:
            return f"{city}/{uf}"
    if any(
        marker in normalized_key
        for marker in [
            "tribunal de justica",
            "tribunal regional eleitoral",
            "tribunal superior eleitoral",
            "jurisprudencia",
            "decisoes",
            "decisões",
            "municipal de",
            "juizo eleitoral",
            "juízo eleitoral",
            "zona eleitoral",
        ]
    ):
        uf = extract_uf_from_text(value)
        return UF_CAPITALS.get(uf, "") if uf else ""
    return value


def normalize_eleicao_value(value: str) -> str:
    if not value:
        return ""
    value = normalize_text(value).strip()
    lowered = normalize_class_text(value)
    if lowered in {"nao especificada", "nao aplicavel", "geral"}:
        return ""
    match = re.search(r"\b(20\d{2})\b", value)
    if match:
        return match.group(1)
    return value


def normalize_tre(value: str, uf: str) -> str:
    if value:
        match = re.search(r"\bTRE-([A-Z]{2})\b", value)
        if match:
            return f"TRE-{match.group(1).upper()}"
        if normalize_class_text(value) == "tse":
            return "TSE"
    if uf:
        return f"TRE-{uf}"
    return value.strip()


def normalize_ministro_name(name: str) -> str:
    raw_name = str(name or "").strip()
    if not raw_name:
        return ""
    split_candidates = [part.strip() for part in re.split(r"\s*[;,]\s*", raw_name) if part.strip()]
    if len(split_candidates) > 1:
        preferred_markers = ("sucessor", "substituto", "atual", "novo relator", "nova relatora")
        for part in split_candidates:
            if any(marker in normalize_class_text(part) for marker in preferred_markers):
                return normalize_ministro_name(part)
        non_original_candidates = [
            part for part in split_candidates if "original" not in normalize_class_text(part)
        ]
        if len(non_original_candidates) == 1:
            return normalize_ministro_name(non_original_candidates[0])
        return normalize_ministro_name(split_candidates[0])

    name = re.sub(r"\[\[.*?\]\]", "", name).strip()
    name = re.sub(r"\[[^\]]+\]\([^)]+\)", "", name).strip()
    name = name.replace("*", "")
    name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
    name = re.sub(r"[\[\]]", "", name).strip()
    name = name.replace("(", "").replace(")", "").strip()
    name = name.rstrip(".;:,").strip()
    if not name:
        return ""
    name = re.sub(r"^(e|E)\s+", "", name).strip()
    name = re.sub(r"^Ministr[oa]s?\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^Ministra\s+", "Min. ", name, flags=re.IGNORECASE)
    name = re.sub(r"^Ministro\s+", "Min. ", name, flags=re.IGNORECASE)
    if not name.startswith("Min."):
        name = f"Min. {name}"
    normalized_key = normalize_class_text(re.sub(r"^Min\.\s*", "", name))
    if re.search(r"\b(kassio|cassio)\b", normalized_key):
        return "Min. Nunes Marques"
    if normalized_key in {"nao especificado", "relator", "ministro", "ministra"}:
        return ""
    alias = MINISTRO_ALIAS_MAP.get(normalized_key)
    if alias:
        return alias
    return name


def normalize_pedido_vista_name(value: str) -> str:
    value = re.sub(r"(?i)^\s*(relator(?:a)?|presidente|vice-?presidente)\s+", "", value).strip()
    name = normalize_ministro_name(value)
    name = re.sub(r"^Min\.\s+(?:Presidente|Vice-Presidente|Relator(?:a)?)\s+", "Min. ", name)
    name = re.sub(r"^Min\.\s+Ministro\s+", "Min. ", name)
    name = re.sub(r"^Min\.\s+Ministra\s+", "Min. ", name)
    if name in {"Min. Presidente", "Min. Relator", "Min. Relatora", "Min. Ministra", "Min. Ministro"}:
        return ""
    return name


def normalize_pedido_vista_value(value: str) -> str:
    return normalize_pedido_vista_name(value) if value else ""


def is_plausible_ministro_name(name: str) -> bool:
    canonical = normalize_ministro_name(name)
    if not canonical:
        return False
    bare = re.sub(r"^Min\.\s*", "", canonical).strip()
    normalized = normalize_class_text(bare)
    if not normalized:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    tokens = normalized.split()
    if len(tokens) < 2 or len(tokens) > 6:
        return False
    if any(token in MINISTRO_INVALID_NAME_TERMS for token in tokens):
        return False
    meaningful = [token for token in tokens if token not in MINISTRO_NAME_PARTICLES]
    if len(meaningful) < 2:
        return False
    return True


def normalize_composicao(value: str) -> str:
    if not value:
        return ""
    value = value.replace(";", ",")
    normalized: list[str] = []
    seen: set[str] = set()
    for part in [p.strip() for p in value.split(",") if p.strip()]:
        if is_mpe_noise_entry(part):
            continue
        name = normalize_ministro_name(part)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    return ", ".join(normalized)


def normalize_ministro_list(values: Iterable[str] | str) -> list[str]:
    if isinstance(values, str):
        raw_values = [part.strip() for part in values.replace(";", ",").split(",") if part.strip()]
    else:
        raw_values = [str(value or "").strip() for value in values if str(value or "").strip()]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if is_mpe_noise_entry(value):
            continue
        name = normalize_ministro_name(value)
        if name and name not in seen:
            seen.add(name)
            normalized.append(name)
    return normalized


def ministro_institutional_group(name: str) -> str:
    normalized = normalize_ministro_name(name)
    if normalized in MINISTROS_STF:
        return "STF"
    if normalized in MINISTROS_STJ:
        return "STJ"
    if normalized in MINISTROS_JURISTAS:
        return "JURISTA"
    return ""


def composicao_institution_counts(values: Iterable[str] | str) -> dict[str, int]:
    counts = {"STF": 0, "STJ": 0, "JURISTA": 0, "DESCONHECIDO": 0}
    for name in normalize_ministro_list(values):
        group = ministro_institutional_group(name) or "DESCONHECIDO"
        counts[group] += 1
    return counts


def composicao_regimental_issue(values: Iterable[str] | str) -> str:
    normalized = normalize_ministro_list(values)
    count = len(normalized)
    if count > 7:
        return "gt7"
    if count < 6:
        return "lt6"
    counts = composicao_institution_counts(normalized)
    if counts["STF"] > 3 or counts["STJ"] > 2 or counts["JURISTA"] > 2:
        return "category_excess"
    # Tolera ate UM unico ministro nao classificado no roster (tipicamente um
    # ministro recem-empossado/substituto, ou uma grafia ainda sem alias): a
    # bancada de 7 continua aproveitavel. So sinaliza quando ha 2+ desconhecidos.
    if count == 7 and counts["DESCONHECIDO"] >= 2:
        return "unknown_institution"
    # So acusa distribuicao incorreta quando TODOS os 7 nomes sao reconhecidos. Com
    # 1 desconhecido nao da para afirmar que a distribuicao esta errada: o nome fora
    # do roster pode justamente ocupar a vaga aparentemente faltante (e qualquer
    # desvio real de (3,2,2) sem desconhecidos ja teria caido em category_excess).
    if count == 7 and counts["DESCONHECIDO"] == 0 and (counts["STF"], counts["STJ"], counts["JURISTA"]) != (3, 2, 2):
        return "distribution"
    return ""


def is_regimentally_valid_composicao(values: Iterable[str] | str) -> bool:
    normalized = normalize_ministro_list(values)
    if len(normalized) != 7:
        return False
    counts = composicao_institution_counts(normalized)
    return (counts["STF"], counts["STJ"], counts["JURISTA"], counts["DESCONHECIDO"]) == (3, 2, 2, 0)


def normalize_classe_processo(value: str) -> str:
    if not value:
        return ""
    normalized = normalize_class_text(value)
    if not normalized:
        return ""
    data = get_canonization_data()
    canon = data["class_norm_map"].get(normalized)
    if canon:
        return canon
    for pattern, canon_value in CLASSE_PROCESSO_MAP:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return canon_value
    if "agravo regimental" in normalized and "agravo em recurso especial" in normalized:
        return "AgRg-AREspe"
    if "agravo regimental" in normalized and "recurso especial" in normalized:
        return "AgRg-REspe"
    if "criacao de zona eleitoral" in normalized or "remanejamento" in normalized:
        return "Czer"
    if "resolucao" in normalized:
        return "PA"
    if "registro de federacao partidaria" in normalized:
        return "RPP"
    if "inquerito" in normalized:
        return "AgR-HC"
    return value.strip()


def identity_overlay_class_key(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw_upper = raw.upper().replace("–", "-").replace("—", "-")
    for prefix, canonical in [("ED-", "ED-"), ("ED ", "ED-"), ("AGRG-", "AgRg-"), ("AGRG ", "AgRg-"), ("AGR-", "AgR-"), ("AGR ", "AgR-")]:
        if raw_upper.startswith(prefix):
            tail = raw[len(prefix) :].strip(" -")
            tail_canon = normalize_classe_processo(tail) or tail.strip()
            return f"{canonical}{tail_canon}".strip("-")
    if raw_upper in {"QO"}:
        return "QO"
    if raw_upper in {"REF-TUTCAUTANT", "REF TUTCAUTANT"}:
        return "Ref-TutCautAnt"
    if raw_upper in {"REF.-MS", "REF-MS", "REF MS"}:
        return "Ref.-MS"

    canonical = normalize_classe_processo(raw)
    if canonical.startswith(("ED-", "AgRg-", "AgR-")) or canonical in {"QO", "Ref-TutCautAnt", "Ref.-MS"}:
        return canonical
    return ""


def normalize_resultado_piece(value: str, classe_processo: str, allowed: Optional[set[str]]) -> str:
    lowered = normalize_class_text(value)
    if not lowered:
        return ""
    data = get_canonization_data()
    direct = data["result_norm_map"].get(lowered)
    if direct and (not allowed or direct in allowed):
        return direct
    if lowered in {
        "em julgamento",
        "julgamento em curso",
        "julgamento em curso aguardando voto vista",
        "pendente de conclusao",
        "nao especificada",
        "sessao encerrada",
        "precedente citado",
    }:
        return ""
    if "suspens" in lowered and "vista" in lowered:
        return "Suspenso por vista"
    if "consulta respondida" in lowered or "respondeu a consulta" in lowered:
        return "Aprovada"
    if "nao conhecid" in lowered:
        return "Não conhecida" if "nao conhecida" in lowered else "Não conhecido"
    if re.search(r"\bprovido\b", lowered) and "nao conhec" in lowered and "desprov" not in lowered:
        return "Provido, Não conhecido"
    if "prejudic" in lowered and "desprov" in lowered:
        return "Prejudicado, Desprovido"
    if re.search(r"proced[eê]n(?:cia|te)\s+parcial|procedente\s+em\s+parte", lowered):
        return "Procedente em parte"
    if re.search(r"parcialmente\s+deferid", lowered):
        return "Parcialmente deferido"
    if "homologad" in lowered:
        return "Aprovada"
    if "parcial provimento" in lowered or "provido em parte" in lowered:
        return "Provido em parte"
    if (
        "negado provimento" in lowered
        or "negou provimento" in lowered
        or "nego provimento" in lowered
        or "nega provimento" in lowered
        or "desprovido" in lowered
        or "não provido" in lowered
        or "nao provido" in lowered
    ):
        return "Desprovido"
    if re.search(r"\bprovido\b|\bprovimento\b", lowered):
        return "Provido"
    if (
        "não conhecido" in lowered
        or "nao conhecido" in lowered
        or "não conhecimento" in lowered
        or "nao conhecimento" in lowered
        or "não conhecer" in lowered
        or "nao conhecer" in lowered
    ):
        return "Não conhecido"
    if re.search(r"aprovad[oa]s?\s+com\s+ressalv", lowered):
        return "Aprovada com ressalvas"
    if "aprov" in lowered:
        return "Aprovada"
    if "indeferid" in lowered:
        return "Indeferida" if "indeferida" in lowered else "Indeferido"
    if "deferid" in lowered:
        return "Deferido"
    if "referendad" in lowered:
        return "Referendada" if "referendada" in lowered else "Referendado"
    if "acolh" in lowered and ("em parte" in lowered or "parcial" in lowered):
        return "Acolhido em parte"
    if "acolhido" in lowered or "acolhidos" in lowered or "acolhida" in lowered:
        return "Acolhidos"
    if "rejeitad" in lowered:
        return "Rejeitada" if "rejeitada" in lowered else "Rejeitados"
    if "devolvid" in lowered:
        return "Devolvida"
    if "prejudicado" in lowered:
        return "Prejudicado"
    return value.strip()


def normalize_resultado_final(value: str, classe_processo: str = "") -> str:
    if not value:
        return ""
    text = normalize_text(value).strip()
    if not text:
        return ""
    data = get_canonization_data()
    classe_canon = normalize_classe_processo(classe_processo) if classe_processo else ""
    allowed = data["class_results"].get(classe_canon, set()) if data else set()
    normalized = normalize_class_text(text)
    canonical_direct = data["result_norm_map"].get(normalized)
    if canonical_direct and (not allowed or canonical_direct in allowed):
        return canonical_direct
    if "suspens" in normalized and ("julgado depois" in normalized or "julgada depois" in normalized or "julgamento posterior" in normalized):
        return "Suspenso mas julgado depois"
    if "suspens" in normalized and "vista" in normalized:
        return "Suspenso por vista"
    if re.search(r"\bprovido\b", normalized) and "nao conhec" in normalized and "desprov" not in normalized:
        return "Provido, Não conhecido"
    if "prejudic" in normalized and "desprov" in normalized:
        return "Prejudicado, Desprovido"

    parts = [part.strip() for part in re.split(r"[;,/]", text) if part.strip()]
    if len(parts) > 1:
        normalized_parts: list[str] = []
        for part in parts:
            part_norm = normalize_resultado_piece(part, classe_canon, allowed)
            if part_norm and part_norm not in normalized_parts:
                normalized_parts.append(part_norm)
        if normalized_parts:
            return ", ".join(normalized_parts)

    single = normalize_resultado_piece(text, classe_canon, allowed)
    return single or text.strip()


def normalize_votacao(value: str) -> str:
    raw_text = str(value or "")
    lowered = normalize_class_text(value)
    if not lowered:
        return ""
    if "suspens" in lowered and "*" in raw_text:
        return "Suspenso*"
    if lowered in {"nao especificada", "em curso"}:
        return ""
    if "unanim" in lowered or "unanime" in lowered:
        return "Unânime"
    if "maioria" in lowered:
        return "Por maioria"
    if re.search(r"\b\d+\s*a\s*\d+\b", lowered):
        return "Por maioria"
    if "julgamento conjunto" in lowered:
        return ""
    if "pedido de vista" in lowered:
        return "Suspenso"
    if "suspens" in lowered:
        return "Suspenso"
    return value.strip()


def parse_date_from_text(text: str) -> str:
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if match:
        a, b, year = map(int, match.groups())
        mdy_ok = True
        dmy_ok = True
        try:
            date(year, a, b)
        except ValueError:
            mdy_ok = False
        try:
            date(year, b, a)
        except ValueError:
            dmy_ok = False
        if a > 12 and dmy_ok:
            return f"{b}/{a}/{year}"
        if b > 12 and mdy_ok:
            return f"{a}/{b}/{year}"
        if dmy_ok:
            return f"{b}/{a}/{year}"
        if mdy_ok:
            return f"{a}/{b}/{year}"
        return ""

    months = {
        "janeiro": "01",
        "fevereiro": "02",
        "março": "03",
        "marco": "03",
        "abril": "04",
        "maio": "05",
        "junho": "06",
        "julho": "07",
        "agosto": "08",
        "setembro": "09",
        "outubro": "10",
        "novembro": "11",
        "dezembro": "12",
    }
    match = re.search(
        r"\b(\d{1,2})\s+de\s+([A-Za-zçÇãõéíóúÁÉÍÓÚ]+)\s+de\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    day, month_name, year = match.groups()
    month = months.get(month_name.lower())
    if not month:
        return ""
    day_int = int(day)
    month_int = int(month)
    year_int = int(year)
    try:
        date(year_int, month_int, day_int)
    except ValueError:
        return ""
    return f"{month_int}/{day_int}/{year_int}"


def normalize_session_date_to_iso(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    parsed = parse_date_from_text(value)
    if not parsed:
        return ""
    month, day, year = [int(part) for part in parsed.split("/")]
    return f"{year:04d}-{month:02d}-{day:02d}"


def split_csv_like_text(value: str) -> list[str]:
    if not value:
        return []
    parts = [item.strip() for item in re.split(r"\s*,\s*|\s*;\s*", value) if item.strip()]
    return dedupe_preserve_order(parts)


def parse_multi_value_text(value: str) -> list[str]:
    if isinstance(value, list):
        return dedupe_preserve_order(str(item).strip() for item in value if str(item).strip())
    return split_csv_like_text(str(value or ""))


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/", 1)[0]
    if "youtube.com" in host:
        query = parse_qs(parsed.query)
        video_id = query.get("v", [""])[0]
        if video_id:
            return video_id
        # /shorts/<id>, /live/<id> (lives das sessoes do TSE), /embed/<id>
        for prefix in ("/shorts/", "/live/", "/embed/"):
            if parsed.path.startswith(prefix):
                return parsed.path.split(prefix, 1)[1].split("/", 1)[0]
    return ""


def normalize_youtube_link(value: str) -> str:
    if not value:
        return ""
    video_id = extract_youtube_video_id(value)
    if not video_id:
        return value.strip()
    parsed = urlparse(value.strip())
    query = parse_qs(parsed.query)
    t_value = query.get("t", [""])[0] or query.get("start", [""])[0]
    if not t_value and parsed.fragment.startswith("t="):
        t_value = parsed.fragment.split("=", 1)[1]
    params = {"v": video_id}
    if t_value:
        params["t"] = t_value
    return f"https://www.youtube.com/watch?{urlencode(params)}"


def build_video_only_youtube_link(value: str) -> str:
    normalized = normalize_youtube_link(value)
    video_id = extract_youtube_video_id(normalized)
    if not video_id:
        return value.strip()
    return f"https://www.youtube.com/watch?{urlencode({'v': video_id})}"


def build_timestamped_youtube_link(value: str, start_seconds: int | None) -> str:
    normalized = build_video_only_youtube_link(value)
    video_id = extract_youtube_video_id(normalized)
    if not video_id:
        return value.strip()
    params = {"v": video_id}
    if start_seconds is not None and start_seconds >= 0:
        params["t"] = str(int(start_seconds))
    return f"https://www.youtube.com/watch?{urlencode(params)}"
