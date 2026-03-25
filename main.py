import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

import rag

console = Console()


def main():
    # Recibir el PDF como argumento
    if len(sys.argv) < 2:
        console.print("[red]Uso: python main.py documento.pdf[/red]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        console.print(f"[red]Archivo no encontrado: {pdf_path}[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]RAG PDF[/bold cyan] — {pdf_path.name}\n")

    # Indexar solo si no existe la colección en ChromaDB
    if rag.is_indexed(str(pdf_path)):
        console.print("[dim]Índice existente encontrado, cargando...[/dim]")
        vectordb = rag.load_index(str(pdf_path))
    else:
        console.print("[dim]Indexando PDF, espera un momento...[/dim]")
        vectordb = rag.index_pdf(str(pdf_path))
        console.print("[green]PDF indexado[/green]")

    qa = rag.build_qa(vectordb)
    console.print("[dim]Escribe 'salir' para terminar.[/dim]\n")

    # Bucle de preguntas
    while True:
        try:
            question = Prompt.ask("[bold]Pregunta[/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question or question.lower() in ("salir", "exit", "q"):
            break

        with console.status("Pensando..."):
            answer, sources = rag.ask(qa, question)

        # Respuesta
        console.print()
        console.print(Markdown(answer))

        # Fuentes (páginas usadas)
        if sources:
            pages = sorted({
                str(d.metadata.get("page", "?") + 1)
                for d in sources
                if isinstance(d.metadata.get("page"), int)
            })
            console.print(f"\n[dim]Páginas consultadas: {', '.join(pages)}[/dim]")

        console.print()

    console.print("[dim]¡Hasta luego![/dim]\n")


if __name__ == "__main__":
    main()