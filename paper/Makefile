# nano ~/.latexmkrc
# $pdf_previewer = 'evince %O %S';

# ieee:
#   	latexmk ieee/ieee.tex -pdf -output-directory=out/ -pvc -interaction=nonstopmode -f -cd
OUTDIR ?= build
LATEXMK = latexmk -f -pdf -output-directory=$(CURDIR)/$(OUTDIR) -pvc -interaction=nonstopmode -cd

TARGET ?= preprint
SRC = $(TARGET)/$(TARGET).tex
PDF = $(OUTDIR)/$(TARGET).pdf

$(PDF): $(SRC)
	mkdir -p $(OUTDIR)
	$(LATEXMK) $(SRC)

all: $(PDF)


clean:
	latexmk -C -output-directory=$(CURDIR)/$(OUTDIR) $(SRC)
	rm -rf $(OUTDIR)