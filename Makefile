LATEXMK ?= latexmk
MAIN ?= main.tex
PDF := $(MAIN:.tex=.pdf)
LOCALE_ENV ?= LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

.PHONY: help pdf watch clean distclean

help:
	@echo "Targets:"
	@echo "  make pdf        Build $(PDF)"
	@echo "  make watch      Rebuild on file changes"
	@echo "  make clean      Remove intermediate files"
	@echo "  make distclean  Remove intermediates and PDF"

pdf:
	$(LOCALE_ENV) $(LATEXMK) -pdf -interaction=nonstopmode -file-line-error $(MAIN)

watch:
	$(LOCALE_ENV) $(LATEXMK) -pvc -pdf -interaction=nonstopmode -file-line-error $(MAIN)

clean:
	$(LOCALE_ENV) $(LATEXMK) -c $(MAIN)

distclean:
	$(LOCALE_ENV) $(LATEXMK) -C $(MAIN)
	rm -f $(PDF)
