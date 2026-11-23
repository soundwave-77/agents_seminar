import datetime

import agents


@agents.function_tool
def get_current_date() -> str:
    """Возвращает текущую местную дату в формате ГГГГ-ММ-ДД."""
    return datetime.date.today().isoformat()
