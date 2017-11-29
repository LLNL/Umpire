if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif ()

if (SPHINX_FOUND)
  set (SPHINX_HTML_THEME "import sphinx_rtd_theme\n
html_theme = \"sphinx_rtd_theme\"\n
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]")

  add_subdirectory(sphinx)
endif ()
