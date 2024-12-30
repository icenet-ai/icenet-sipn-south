.PHONY: html clean open

html:
	$(MAKE) -C docs html

clean:
	$(MAKE) -C docs clean

open:
	xdg-open docs/build/html/index.html

