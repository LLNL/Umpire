#!/usr/bin/env python3

from lxml import etree
import sys

xmlfile = open(sys.argv[1], 'r')
xslfile = open(sys.argv[2], 'r')

xslcontent = xslfile.read()
xmldoc = etree.parse(xmlfile)
xslt_root = etree.XML(xslcontent)
transform = etree.XSLT(xslt_root)

result_tree = transform(xmldoc)
print(result_tree)
