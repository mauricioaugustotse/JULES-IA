#!/usr/bin/env bash
set -euo pipefail

mode="${1:-commit}"

blocked_path_regex='(^|/)(\.env(\..*)?|Chave.*\.txt|chave.*\.txt|CHAVE.*\.txt|.*[Nn][Oo][Tt][Ii][Oo][Nn].*\.txt|.*[Oo][Pp][Ee][Nn][Aa][Ii].*\.txt|.*[Pp][Ee][Rr][Pp][Ll][Ee][Xx][Ii][Tt][Yy].*\.txt|.*[Tt][Oo][Kk][Ee][Nn].*\.txt|.*\.(key|pem|p12))$|(^|/)\.secrets(/|$)'
secret_value_regex='(secret_[A-Za-z0-9]{30,}|ntn_[A-Za-z0-9_-]{30,}|sk-[A-Za-z0-9_-]{30,}|AIza[0-9A-Za-z_-]{35})'

fail=0

if [[ "$mode" == "commit" ]]; then
    while IFS= read -r path; do
        [[ -z "$path" ]] && continue
        case "$path" in
            .env.example|.env.sample|.env.template) continue ;;
        esac
        if [[ "$path" =~ $blocked_path_regex ]]; then
            echo "Bloqueado: arquivo com nome de segredo esta staged: $path" >&2
            fail=1
        fi
    done < <(git diff --cached --name-only --diff-filter=ACMR)

    if git diff --cached --unified=0 --no-ext-diff | grep -E "^\+[^+].*$secret_value_regex" >/dev/null; then
        echo "Bloqueado: o diff staged contem valor com aparencia de chave/API token." >&2
        fail=1
    fi
else
    if git grep -I -l -E "$secret_value_regex" -- . ':!.githooks/*' >/dev/null; then
        echo "Bloqueado: arquivos rastreados contem valor com aparencia de chave/API token." >&2
        echo "Use: git grep -I -l -E '$secret_value_regex' -- . ':!.githooks/*'" >&2
        fail=1
    fi
fi

if [[ "$fail" -ne 0 ]]; then
    echo "Mova chaves para arquivos ignorados, como Chave_Notion.txt, ou para .env local nao versionado." >&2
    exit 1
fi
