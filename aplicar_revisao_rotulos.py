
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib

BASE = Path("labels_revisados/train")
ORIGINAIS = Path("labels_backup_original/train")

REVISAO = Path("auditoria_rotulos/revisao_manual.csv")
RELATORIO = Path("auditoria_rotulos/relatorio_correcao.csv")

CLASSE_ALVO = "1"  # backpack_bag


def hash_texto(texto):
    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Corrige duplicacoes confirmadas manualmente."
    )
    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="Aplica as correcoes apos a simulacao."
    )
    args = parser.parse_args()

    # Agrupa as decisoes confirmadas por arquivo.
    decisoes = defaultdict(list)

    with REVISAO.open(
        "r", encoding="utf-8-sig", newline=""
    ) as arquivo:
        leitor = csv.DictReader(arquivo)

        campos = {
            "arquivo",
            "linha_mantida",
            "linha_candidata_remocao",
            "decisao"
        }

        if not campos.issubset(leitor.fieldnames or []):
            raise ValueError(
                "O CSV de revisao nao possui os campos esperados."
            )

        for registro in leitor:
            if registro["decisao"].strip().lower() != "sim":
                continue

            nome = Path(registro["arquivo"]).name

            if nome != registro["arquivo"]:
                raise ValueError(
                    "Nome de arquivo invalido: " + registro["arquivo"]
                )

            manter = int(registro["linha_mantida"])
            remover = int(registro["linha_candidata_remocao"])

            if manter == remover:
                raise ValueError(
                    "Linha mantida e removida sao iguais: " + nome
                )

            decisoes[nome].append((manter, remover))

    print("\n=== CORRECAO DE ROTULOS ===")
    print("Modo:", "APLICACAO" if args.aplicar else "SIMULACAO")
    print("Arquivos selecionados:", len(decisoes))

    planos = []
    relatorio = []
    erros = []

    for nome, pares in sorted(decisoes.items()):
        destino = BASE / nome
        origem = ORIGINAIS / nome

        if not destino.is_file() or not origem.is_file():
            erros.append(f"Arquivo ausente: {nome}")
            continue

        # A numeracao das linhas refere-se ao backup original.
        texto_original = origem.read_text(encoding="utf-8")
        texto_atual = destino.read_text(encoding="utf-8")

        if hash_texto(texto_original) != hash_texto(texto_atual):
            erros.append(
                f"Arquivo revisado diferente do backup: {nome}"
            )
            continue

        linhas = texto_atual.splitlines()
        numeros_remover = set()
        valido = True

        for manter, remover in pares:
            if not (
                1 <= manter <= len(linhas)
                and 1 <= remover <= len(linhas)
            ):
                erros.append(f"Numero de linha invalido: {nome}")
                valido = False
                break

            dados_mantidos = linhas[manter - 1].split()
            dados_removidos = linhas[remover - 1].split()

            if (
                len(dados_mantidos) != 5
                or len(dados_removidos) != 5
                or dados_mantidos[0] != CLASSE_ALVO
                or dados_removidos[0] != CLASSE_ALVO
            ):
                erros.append(
                    f"Classe ou formato inesperado: {nome}"
                )
                valido = False
                break

            if remover in numeros_remover:
                erros.append(
                    f"Linha repetida para remocao: {nome}"
                )
                valido = False
                break

            numeros_remover.add(remover)

            relatorio.append({
                "arquivo": nome,
                "linha_mantida": manter,
                "linha_removida": remover,
                "conteudo_removido": linhas[remover - 1]
            })

        if not valido:
            continue

        if any(
            manter in numeros_remover
            for manter, _ in pares
        ):
            erros.append(
                f"Uma linha mantida tambem seria removida: {nome}"
            )
            continue

        novas_linhas = [
            linha
            for numero, linha in enumerate(linhas, start=1)
            if numero not in numeros_remover
        ]

        novo_texto = "\n".join(novas_linhas)

        if novas_linhas:
            novo_texto += "\n"

        planos.append((destino, novo_texto))

    print("Arquivos validados:", len(planos))
    print("Anotacoes candidatas:", len(relatorio))
    print("Erros encontrados:", len(erros))

    if erros:
        print("\nERROS:")
        for erro in erros[:20]:
            print("-", erro)

        print("\nOPERACAO CANCELADA.")
        print("Nenhum rotulo foi modificado.")
        return

    if not args.aplicar:
        print("\nSIMULACAO CONCLUIDA.")
        print("Nenhum rotulo foi modificado.")
        print(
            "Para aplicar posteriormente, execute com --aplicar."
        )
        return

    # Registra o que sera alterado antes de escrever nos arquivos.
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)

    with RELATORIO.open(
        "w", encoding="utf-8-sig", newline=""
    ) as arquivo:
        campos = [
            "arquivo",
            "linha_mantida",
            "linha_removida",
            "conteudo_removido"
        ]

        escritor = csv.DictWriter(
            arquivo, fieldnames=campos
        )
        escritor.writeheader()
        escritor.writerows(relatorio)

    for destino, novo_texto in planos:
        destino.write_text(
            novo_texto, encoding="utf-8"
        )

    print("\nCORRECAO APLICADA.")
    print("Arquivos modificados:", len(planos))
    print("Anotacoes removidas:", len(relatorio))
    print("Relatorio:", RELATORIO.resolve())
    print("\nBackup original preservado.")


if __name__ == "__main__":
    main()
