# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import shutil
import jinja2
import os

# Environment to process Jinja templates.
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader("."))

# Jinja templates to render out.
templates = []

# Render templates and output files without the last extension.
# For example: 'install.md.jinja' becomes 'install.md'.
for template in templates:
    rendered = jinja_env.get_template(template).render()
    with open(os.path.splitext(template)[0], "w") as file:
        file.write(rendered)

# shutil.copy2("../RELEASE.md", "./about/release-notes.md")
# # Keep capitalization due to similar linking on GitHub's markdown preview.
# shutil.copy2("../CHANGELOG.md", "./about/CHANGELOG.md")

latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand\ttdefault{txtt}
"""
}

# configurations for PDF output by Read the Docs
project = "ROCm Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved."
version = "6.0.1"
release = "6.0.1"
setting_all_article_info = True
all_article_info_os = ["linux", "windows"]
all_article_info_author = ""

# pages with specific settings
# article_pages = [
#     {"file": "about/release-notes", "os": ["linux", "windows"], "date": "2024-01-31"},
#     {"file": "about/CHANGELOG", "os": ["linux", "windows"], "date": "2024-01-31"},
#     {"file": "install/windows/install-quick", "os": ["windows"]},
#     {"file": "install/linux/install-quick", "os": ["linux"]},
#     {"file": "install/linux/install", "os": ["linux"]},
#     {"file": "install/linux/install-options", "os": ["linux"]},
#     {"file": "install/linux/prerequisites", "os": ["linux"]},
#     {"file": "install/docker", "os": ["linux"]},
#     {"file": "install/magma-install", "os": ["linux"]},
#     {"file": "install/pytorch-install", "os": ["linux"]},
#     {"file": "install/tensorflow-install", "os": ["linux"]},
#     {"file": "install/windows/install", "os": ["windows"]},
#     {"file": "install/windows/prerequisites", "os": ["windows"]},
#     {"file": "install/windows/cli/index", "os": ["windows"]},
#     {"file": "install/windows/gui/index", "os": ["windows"]},
#     {"file": "about/compatibility/docker-image-support-matrix", "os": ["linux"]},
#     {"file": "about/compatibility/user-kernel-space-compat-matrix", "os": ["linux"]},
#     {"file": "reference/library-index", "os": ["linux"]},
#     {"file": "how-to/deep-learning-rocm", "os": ["linux"]},
#     {"file": "how-to/gpu-enabled-mpi", "os": ["linux"]},
#     {"file": "how-to/system-debugging", "os": ["linux"]},
#     {"file": "how-to/tuning-guides", "os": ["linux", "windows"]},
#     {"file": "rocm-a-z", "os": ["linux", "windows"]},
# ]

exclude_patterns = ["temp"]

external_toc_path = "./sphinx/_toc.yml"

extensions = ["rocm_docs"]

external_projects_current_project = "rocm"

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-docs-home"}

html_title = "ROCm Documentation"

html_theme_options = {"link_main_doc": False}
