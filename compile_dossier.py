import os
import glob

def compile_markdown():
    print("Compiling markdown sections...")
    sections = sorted(glob.glob("dossier_sections/*.md"))
    
    with open("PhysioFM_Research_Dossier.md", "w", encoding="utf-8") as outfile:
        # Title Page
        outfile.write("---\\n")
        outfile.write("title: 'PhysioFM: Optimization Diagnostics and Failure Analysis of Hierarchical Foundation Models in rPPG'\\n")
        outfile.write("author: 'Lead Research Scientist'\\n")
        outfile.write("date: 'July 2026'\\n")
        outfile.write("---\\n\\n")
        
        for section in sections:
            with open(section, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\\n\\n---\\n\\n")
    print("Markdown compiled: PhysioFM_Research_Dossier.md")

if __name__ == "__main__":
    compile_markdown()
