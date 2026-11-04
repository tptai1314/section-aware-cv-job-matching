# Academic Paper: Section-Aware CV-Job Matching

Bản thảo bài báo khoa học nghiên cứu ứng dụng (Applied Paper) theo định dạng hội nghị quốc tế **Springer LNCS** (Lecture Notes in Computer Science).

## Cấu trúc thư mục

```
paper/
├── main.tex                    # Master LaTeX document (tổng hợp các phần)
├── main.pdf                    # File PDF bài báo hoàn chỉnh (sau khi biên dịch)
├── references.bib              # Danh mục tài liệu tham khảo (BibTeX chuẩn)
├── llncs.cls                   # Document class Springer LNCS
├── splncs04.bst                # BibTeX style Springer LNCS
├── build.ps1                   # Script tự động biên dịch LaTeX 1-click
├── figures/                    # Hình vẽ kiến trúc & sơ đồ độ phân giải cao
│   ├── fig01_tong_the.png
│   ├── fig07_vector_theo_section.png
│   └── fig08_luong_ghep_cap.png
├── tables/                     # Các bảng số liệu thực nghiệm tách riêng
│   ├── tab_main_results.tex    # Bảng 1: Kết quả xếp hạng đối đầu 8 cấu hình
│   └── tab_alpha_sweep.tex     # Bảng 2: Phân tích độ nhạy tham số Alpha
└── sections/                   # Nội dung từng phần của bài báo
    ├── 00_abstract.tex         # Tóm tắt & từ khóa
    ├── 01_introduction.tex     # Đặt vấn đề & đóng góp
    ├── 02_related_work.tex     # Khảo cứu công trình liên quan
    ├── 03_methodology.tex      # Phương pháp, biểu diễn đa vector & Hybrid
    ├── 04_experiments.tex      # Tập dữ liệu & thiết lập thực nghiệm
    ├── 05_results.tex          # Kết quả, phân tích số liệu & Case study
    └── 06_conclusion.tex       # Kết luận & hướng phát triển
```

## Cách biên dịch ra PDF

### Cách 1: Dùng script 1-click (PowerShell)
```powershell
cd section-aware-cv-job-matching/paper
./build.ps1
```

### Cách 2: Dùng lệnh pdflatex thủ công
```powershell
cd section-aware-cv-job-matching/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

