class ApiError(Exception):
    """Lỗi trả ra ngoài theo khuôn {detail, message} của spec mục 3.3."""

    def __init__(self, status: int, detail: str, message: str):
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.message = message


def cv_file_required() -> ApiError:
    return ApiError(400, "CV_FILE_REQUIRED", "Thiếu field `file` trong multipart request")


def cv_file_invalid(detail: str = "CV_FILE_INVALID") -> ApiError:
    return ApiError(400, detail, "File không phải PDF/DOCX hợp lệ hoặc magic bytes sai")


def cv_file_too_large() -> ApiError:
    return ApiError(413, "CV_FILE_TOO_LARGE", "File vượt quá dung lượng cho phép")


def cv_text_empty() -> ApiError:
    return ApiError(422, "CV_TEXT_EMPTY", "Không trích xuất được nội dung từ file")


def cv_encrypted() -> ApiError:
    return ApiError(422, "CV_ENCRYPTED", "PDF có mật khẩu / bị mã hoá")


def llm_unavailable() -> ApiError:
    return ApiError(500, "LLM_UNAVAILABLE", "Không kết nối được Ollama")


def llm_invalid_output() -> ApiError:
    return ApiError(500, "LLM_INVALID_OUTPUT", "Model trả JSON không hợp lệ sau khi retry")


def unauthorized() -> ApiError:
    return ApiError(401, "UNAUTHORIZED", "Thieu hoac sai internal token")


def llm_timeout() -> ApiError:
    return ApiError(504, "LLM_TIMEOUT", "Xử lý vượt quá thời gian cho phép")


def job_not_found() -> ApiError:
    return ApiError(404, "JOB_NOT_FOUND", "Không tìm thấy job xử lý theo id này")


def job_conflict(detail: str, message: str) -> ApiError:
    return ApiError(409, detail, message)


def bad_request(detail: str, message: str) -> ApiError:
    return ApiError(400, detail, message)


def invalid_document_type() -> ApiError:
    return ApiError(422, "INVALID_DOCUMENT_TYPE", "`document_type` phải là `cv` hoặc `jd`")


def job_source_required(detail: str = "JOB_SOURCE_REQUIRED", message: str | None = None) -> ApiError:
    return ApiError(
        422,
        detail,
        message or "Job thiếu nguồn nội dung: cần `file` (upload) hoặc `text` (đã trích xuất)",
    )