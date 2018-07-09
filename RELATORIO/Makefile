# Makefile para relatório 


LATEX = pdflatex --shell-escape

REPORT = bare_conf.pdf

DEPS =  bare_conf.tex \
	bare_conf.bib

all: $(REPORT)
	
$(REPORT) : $(DEPS)
	$(LATEX) $<
	echo $(basename $<)
	biber $(basename $<)
	$(LATEX) $<
	$(LATEX) $<
	echo "compilação concluída com sucesso"

.PHONY: clean

clean:
	rm *.pdf
	rm *.log
	rm *.aux
	rm *.blg
	rm *.bbl
	rm *.run.xml

read:
	evince $(REPORT)
