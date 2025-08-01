import sys
import os
from rich.console import Console

def main():
    arg = sys.argv  
    console = Console()
    content = None

    if "?" in arg or "--help" in arg:
        console.print("[cyan bold]Usage:[/]\n  --file <filename>\n  --input <text>\n\nNote: If your filename or input contains spaces, wrap it in quotes.")
        sys.exit(0)

    if '--file' in arg:
        try:
            file_name = arg[arg.index('--file') + 1] 
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"File '{file_name}' not found.")
            with open(file_name, 'r') as file:
                content = file.read()
        except IndexError:
            console.print("[red bold]Error:[/] No file provided after '--file'")
            sys.exit(1)
        except FileNotFoundError as e:
            console.print(f"[red bold]Error:[/] {str(e)}")
            sys.exit(1)

    elif '--input' in arg:
        try:
            content = arg[arg.index('--input') + 1]
            if not content.strip():
                raise ValueError()
        except (IndexError, ValueError):
            console.print("[red bold]Error:[/] No text provided after '--input'")
            sys.exit(1)

    else:
        console.print("[yellow]No arguments passed. Waiting for input from stdin (press Ctrl+D when done):[/]")
        content = sys.stdin.read()

    console.print("[green bold]Content loaded successfully.[/]")

if __name__ == '__main__':
    main()
