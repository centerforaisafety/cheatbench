# Office task dependencies, built automatically on a fresh checkout.
# This image contains no task data, grader, model runtime, or credentials.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git libreoffice-writer libreoffice-calc \
        libreoffice-impress fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
        numpy sympy mpmath matplotlib pillow==12.3.0 requests==2.34.2 \
        pyyaml==6.0.3 PyMuPDF python-docx python-pptx openpyxl pypdf reportlab
RUN python -c "import fitz, PIL, numpy, docx, pptx, openpyxl" \
    && soffice --headless --version
RUN mkdir -p /workspace
WORKDIR /workspace
