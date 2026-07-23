"""Evidence-grounded weekly overdue-report summarisation route."""

from fastapi import APIRouter, Request
from slowapi import Limiter

from src.api.errors import ErrorResponse
from src.api.middleware import tenant_rate_limit_key
from src.api.models.requests.weekly_report import WeeklyOverdueReportSummaryRequest
from src.api.models.responses import WeeklyOverdueReportSummaryResponse
from src.config.settings import settings
from src.engine.weekly_overdue_report_summarizer import weekly_overdue_report_summarizer

router = APIRouter()
limiter = Limiter(key_func=tenant_rate_limit_key)


@router.post(
    "/summarize-weekly-overdue-report",
    response_model=WeeklyOverdueReportSummaryResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
@limiter.limit(settings.rate_limit_classify)
async def summarize_weekly_overdue_report(
    request: Request,
    summary_request: WeeklyOverdueReportSummaryRequest,
) -> WeeklyOverdueReportSummaryResponse:
    return await weekly_overdue_report_summarizer.summarize(summary_request)
