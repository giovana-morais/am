# Makefile para relatório 


LATEX = xelatex --shell-escape

REPORT = bare_conf.pdf

DEPS = bare_conf.tex \
	#IEEEtranBST/IEEEfull.bib

all: $(REPORT)
	
$(REPORT) : $(DEPS)
	$(LATEX) $<
	echo $(basename $<)
#	biber $(basename $<)
	$(LATEX) $<
	$(LATEX) $<

.PHONY: clean

clean:
	rm *.pdf
	rm *.log
	rm *.aux

read:
	evince $(REPORT)
