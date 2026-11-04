---
pretty_name: Vietnamese Job Descriptions
language:
- vi
license: cc-by-nc-4.0
task_categories:
- text-classification
- feature-extraction
tags:
- vietnam
- vietnamese
- job-description
- recruitment
- labor-market
- career
- salary-prediction
- skill-extraction
- education
- experience
- tabular
- text
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data.parquet
---


# 💼 Tinix Vietnam Job Description

## 1. 📌 Giới Thiệu Tinix Vietnam Job Description

**Tinix Vietnam Job Description** là bộ dữ liệu tuyển dụng tiếng Việt ở định dạng CSV, gồm các tin tuyển dụng có cấu trúc về chức danh, công ty, mức lương, địa điểm, loại hợp đồng, ngành nghề, yêu cầu kinh nghiệm, trình độ học vấn, mô tả công việc, phúc lợi, yêu cầu ứng viên và năm đăng tin.

Bộ dữ liệu được thiết kế cho các bài toán NLP và phân tích thị trường lao động tại Việt Nam, đặc biệt trong miền tuyển dụng, HR analytics và job intelligence.

**Điểm nổi bật**

- 606,878 bản ghi tin tuyển dụng.
- 14 cột dữ liệu có cấu trúc.
- Bao phủ các năm 2022, 2023, 2024, 2025 và 2026.
- Nội dung chính bằng tiếng Việt, có thể kèm thuật ngữ tiếng Anh, viết tắt và ngôn ngữ đặc thù của từng công ty.

## 2. 📊 Market Overview & Statistical Insight

### Tổng quan nhanh

| Chỉ số | Giá trị |
| --- | ---: |
| Số bản ghi | 606,878 |
| Số cột | 14 |
| Giai đoạn dữ liệu | 2022-2026 |
| Số `id` duy nhất | 606,878 |
| Số `job_title` duy nhất | 299,021 |
| Số `company_name` duy nhất | 115,570 |
| Số `job_industry` duy nhất | 4,358 |
| Số `location` duy nhất | 250,580 |

### Phân bố theo năm

![Phân bố theo năm](figures/records_by_year.png)



### Yêu cầu kinh nghiệm phổ biến

![Phân bố kinh nghiệm](figures/experience_distribution.png)


### Trình độ học vấn phổ biến

![Phân bố trình độ học vấn](figures/education_distribution.png)


### Ngành nghề nổi bật

| Ngành nghề | Số bản ghi |
| --- | ---: |
| Bán hàng - Kinh doanh | 58,932 |
| Chăm sóc khách hàng | 56,083 |
| Kế toán / Kiểm toán | 50,242 |
| Xây dựng | 29,757 |
| Chưa xác định | 25,638 |
| Cơ khí - Ô tô - Tự động hóa / Sản xuất - Lắp ráp - Chế biến | 24,861 |
| Khoa học - Kỹ thuật | 24,001 |
| Nghề nghiệp khác | 23,330 |
| Lao động phổ thông | 23,285 |
| Giáo dục - Đào tạo | 18,533 |

## 3. 🧩 Data Schema

| Trường | Kiểu dữ liệu | Mô tả |
| --- | --- | --- |
| `id` | integer | Mã định danh duy nhất của tin tuyển dụng. |
| `job_title` | string | Chức danh công việc theo nội dung gốc. |
| `company_name` | string | Tên công ty đăng tuyển. |
| `salary` | string | Thông tin lương dạng văn bản. |
| `location` | string | Địa điểm làm việc dạng văn bản. |
| `job_type` | string | Loại hợp đồng hoặc hình thức làm việc, ví dụ toàn thời gian, bán thời gian, thực tập, remote. |
| `job_industry` | string | Ngành nghề hoặc nhóm ngành của tin tuyển dụng. |
| `experience_level` | string | Yêu cầu kinh nghiệm dạng văn bản. |
| `education_level` | string | Yêu cầu trình độ học vấn. |
| `job_position` | string | Cấp bậc/chức vụ, ví dụ nhân viên, trưởng nhóm, trưởng phòng, giám đốc. |
| `job_description` | string | Nội dung mô tả công việc. |
| `benefits` | string | Phúc lợi, đãi ngộ và quyền lợi ứng viên. |
| `requirements` | string | Yêu cầu ứng viên, kỹ năng, kinh nghiệm và điều kiện liên quan. |
| `year` | integer | Năm đăng tin. |



## 4. 🚀 Potential Use Cases

Bộ dữ liệu phù hợp cho các nhóm ứng dụng sau:

- **Phân loại văn bản**: dự đoán ngành nghề, loại hợp đồng, cấp bậc, yêu cầu kinh nghiệm hoặc trình độ học vấn.
- **Trích xuất thông tin**: trích xuất kỹ năng, lương, địa điểm, phúc lợi, bằng cấp và yêu cầu ứng viên.
- **Embedding và tìm kiếm ngữ nghĩa**: huấn luyện/evaluate embedding tiếng Việt trong miền tuyển dụng.
- **Tóm tắt nội dung tuyển dụng**: rút gọn mô tả công việc dài thành job brief ngắn gọn.
- **Sinh nội dung tuyển dụng**: tạo mô tả công việc, quyền lợi, yêu cầu ứng viên theo mẫu.
- **Phân tích thị trường lao động**: theo dõi xu hướng theo năm, ngành nghề, địa điểm, kinh nghiệm và hình thức làm việc.
- **Chuẩn hóa dữ liệu HR**: gom nhóm các giá trị tự do như lương, địa điểm, ngành nghề và cấp bậc.


## 4. 🚀 Potential Use Cases

Bộ dữ liệu này có thể được sử dụng cho nhiều bài toán trong lĩnh vực tuyển dụng, phân tích thị trường lao động và xây dựng hệ thống trí tuệ nghề nghiệp. Do dữ liệu bao gồm cả thông tin từ CV và mô tả công việc, nó phù hợp cho các nhóm ứng dụng sau:

- **Phân loại văn bản tuyển dụng**: dự đoán ngành nghề, loại hợp đồng, cấp bậc nghề nghiệp, yêu cầu kinh nghiệm hoặc trình độ học vấn.
- **Trích xuất thông tin nhân sự**: nhận diện và trích xuất kỹ năng, mức lương, địa điểm, phúc lợi, bằng cấp, yêu cầu ứng viên và mô tả trách nhiệm công việc.
- **Tìm kiếm ngữ nghĩa và embedding tiếng Việt**: huấn luyện hoặc đánh giá các mô hình embedding trong miền tuyển dụng, bao gồm tìm kiếm CV phù hợp với JD và ngược lại.
- **CV--JD matching**: hỗ trợ đánh giá mức độ phù hợp giữa hồ sơ ứng viên và mô tả công việc dựa trên kỹ năng, kinh nghiệm, cấp bậc và yêu cầu tuyển dụng.
- **Tóm tắt nội dung tuyển dụng**: rút gọn mô tả công việc dài thành các bản tóm tắt ngắn gọn, dễ đọc và có cấu trúc.
- **Sinh nội dung tuyển dụng**: hỗ trợ tạo mô tả công việc, yêu cầu ứng viên, quyền lợi và tiêu chí tuyển dụng theo từng vị trí.
- **Phân tích thị trường lao động**: theo dõi xu hướng tuyển dụng theo năm, ngành nghề, địa điểm, cấp bậc, kinh nghiệm và hình thức làm việc.
- **Chuẩn hóa dữ liệu HR**: gom nhóm và chuẩn hóa các trường dữ liệu tự do như kỹ năng, chức danh, địa điểm, mức lương, ngành nghề và cấp bậc nghề nghiệp.
- **Xây dựng đồ thị tri thức nghề nghiệp**: tổ chức các thực thể như kỹ năng, chức danh, cấp bậc và yêu cầu công việc thành một đồ thị tri thức phục vụ suy luận và gợi ý nghề nghiệp.

## 6. 🕸️ Career Path Knowledge Graph

<img src="figures/cpkg_framework.png" alt="CPKG Framework" width="80%">


Bên cạnh các bài toán xử lý dữ liệu dạng bảng và văn bản, bộ dữ liệu này cũng đã được sử dụng để xây dựng một **Career Path Knowledge Graph** nhằm mô hình hóa quan hệ giữa hồ sơ ứng viên, mô tả công việc, kỹ năng, chức danh và cấp bậc nghề nghiệp.

Đồ thị được xây dựng từ dữ liệu CV và job descriptions sau khi chuẩn hóa các trường thông tin quan trọng như kỹ năng, yêu cầu công việc, chức danh và career level. Thay vì chỉ lưu trữ dữ liệu tuyển dụng ở dạng rời rạc, knowledge graph cho phép biểu diễn các mối quan hệ có cấu trúc giữa các thực thể nghề nghiệp, ví dụ như một vị trí yêu cầu những kỹ năng nào, một ứng viên sở hữu những kỹ năng nào, hoặc các chức danh có thể được tổ chức theo những cấp bậc nghề nghiệp nào.

Trên nền tảng đó, dữ liệu có thể hỗ trợ một framework tuyển dụng thông minh cho doanh nghiệp. Framework này hướng tới ba nhóm ứng dụng chính: **đánh giá CV**, **so khớp CV-JD**, và **gợi ý định hướng nghề nghiệp**. Cách tiếp cận dựa trên knowledge graph giúp hệ thống không chỉ dựa vào độ tương đồng văn bản, mà còn khai thác được quan hệ có cấu trúc giữa kỹ năng, vị trí công việc và lộ trình phát triển nghề nghiệp.

### Graph Statistics

| Statistic | Count |
|---|---:|
| Number of CVs | 13,129 |
| Avg. Skills per CV | 15 |
| Avg. Requirements per JD | 8 |
| Distinct Job Titles | 71 |
| Career Levels | 5 |
| Graph Nodes | 355 |
| Graph Edges | 6,254 |
| Average Out-degree | 8.16 |
| Average Skills per Node | 15.10 |


## 7. ⚡ Quickstart

```python
from datasets import load_dataset

dataset = load_dataset(
    "tinixai/vietnamese-job-descriptions",
    split="train",
)

print(dataset)
print(dataset[0])
```

## 8. 📄 License & Citation

Bộ dữ liệu được phát hành theo **Creative Commons Attribution Non-Commercial 4.0 International (CC BY-NC 4.0) license**.

Theo license này, người dùng có thể chia sẻ và điều chỉnh bộ dữ liệu cho mục đích phi thương mại, với điều kiện ghi công phù hợp. Vui lòng kiểm tra điều khoản license và điều khoản nguồn dữ liệu liên quan trước khi sử dụng trong sản phẩm, thương mại hóa hoặc môi trường production.

Nếu sử dụng bộ dữ liệu này, vui lòng trích dẫn:

```bibtex
@inproceedings{le-etal-2026-careerpathkg,
  title     = {{CareerPathKG}: Knowledge Graph Integrated Framework for Career Intelligence},
  author    = {Le, Ngoc-Quang and Hoang, Duc Duong and Tran, Mai Vu and Vuong, Thi-Hai-Yen},
  booktitle = {Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 5: Industry Track)},
  publisher = {Association for Computational Linguistics},
  pages     = {813--822},
  doi       = {10.18653/v1/2026.eacl-industry.60},
  url       = {https://aclanthology.org/2026.eacl-industry.60/}
}
```
