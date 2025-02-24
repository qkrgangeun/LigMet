import os
from pathlib import Path
Fasta_dir = Path('/home/qkrgangeun/LigMet/data/biolip/fasta')
# List all .fasta files in the current directory
fasta_lists = [file for file in os.listdir(Fasta_dir) if file.endswith('.fasta')]

# Create or open the output file
with open(Fasta_dir / 'fasta_input.txt', 'w') as f:
    # Iterate over each .fasta file
    for fasta_file in fasta_lists:
        try:
            # Open the current .fasta file
            with open(Fasta_dir / fasta_file, 'r') as ff:
                # Read the first line from the .fasta file
                context = ff.read()
                # Write the first line to the output file
                f.write(context)
                # Optionally, write a newline character if you want to separate entries clearly
        except Exception as e:
            print(f"Error processing file {fasta_file}: {e}")

