"""
Dịch vụ phân tích dữ liệu giá dầu

Đọc file price_oils.csv, lọc theo khoảng thời gian mô phỏng (start_time / end_time
trong .env), tính toán các chỉ báo kỹ thuật đơn giản và trả về kết quả dạng text
để ReportAgent tool sử dụng.

File CSV có cấu trúc:
  Date, Price, Open, High, Low, Vol., Change %
  "04/30/2026","113.22","111.17","114.68","110.34","469.41K","2.52%"

Dữ liệu được sắp xếp giảm dần (ngày mới nhất ở đầu).
"""

import csv
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.price_data')


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date từ format MM/DD/YYYY hoặc YYYY-MM-DD."""
    date_str = date_str.strip().strip('"')
    for fmt in ('%m/%d/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _parse_float(value: str) -> Optional[float]:
    """Parse float, bỏ qua dấu ngoặc kép và ký tự không hợp lệ."""
    value = value.strip().strip('"')
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_change(value: str) -> Optional[float]:
    """Parse change% string, ví dụ '-0.69%' -> -0.69."""
    value = value.strip().strip('"').replace('%', '')
    return _parse_float(value)


def _parse_volume(value: str) -> str:
    """Giữ nguyên volume string (đã có suffix K/M)."""
    return value.strip().strip('"')


def _load_price_data(csv_path: str = None) -> List[Dict[str, Any]]:
    """
    Đọc toàn bộ file CSV và trả về list các row đã parse.
    Kết quả sắp xếp theo ngày tăng dần (cũ → mới).
    """
    csv_path = csv_path or Config.PRICE_DATA_CSV

    if not os.path.exists(csv_path):
        logger.warning(f"Price data file not found: {csv_path}")
        return []

    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = _parse_date(row.get('Date', ''))
            if dt is None:
                continue

            price = _parse_float(row.get('Price', ''))
            if price is None:
                continue

            rows.append({
                'date': dt,
                'date_str': dt.strftime('%Y-%m-%d'),
                'price': price,
                'open': _parse_float(row.get('Open', '')),
                'high': _parse_float(row.get('High', '')),
                'low': _parse_float(row.get('Low', '')),
                'volume': _parse_volume(row.get('Vol.', '')),
                'change_pct': _parse_change(row.get('Change %', '')),
            })

    # Sắp xếp theo ngày tăng dần
    rows.sort(key=lambda r: r['date'])
    return rows


def _filter_by_time_range(
    rows: List[Dict[str, Any]],
    start_time: str = None,
    end_time: str = None,
) -> List[Dict[str, Any]]:
    """
    Lọc rows theo khoảng thời gian.

    Nếu start_time / end_time không cung cấp, dùng giá trị từ Config.
    Nếu cả hai đều trống, trả về toàn bộ dữ liệu.
    """
    start_str = (start_time or Config.SIMULATION_START_TIME).strip()
    end_str = (end_time or Config.SIMULATION_END_TIME).strip()

    start_dt = _parse_date(start_str) if start_str else None
    end_dt = _parse_date(end_str) if end_str else None

    if start_dt is None and end_dt is None:
        return rows

    filtered = []
    for row in rows:
        if start_dt and row['date'] < start_dt:
            continue
        if end_dt and row['date'] > end_dt:
            continue
        filtered.append(row)

    return filtered


def _compute_sma(prices: List[float], window: int) -> Optional[float]:
    """Simple Moving Average trên window ngày gần nhất."""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window


def _compute_streak(changes: List[Optional[float]]) -> str:
    """Tính chuỗi tăng/giảm liên tục gần nhất."""
    if not changes:
        return "N/A"

    # Lọc bỏ None
    valid = [c for c in reversed(changes) if c is not None]
    if not valid:
        return "N/A"

    direction = "TĂNG" if valid[0] > 0 else ("GIẢM" if valid[0] < 0 else "ĐI NGANG")
    count = 0
    for c in valid:
        if direction == "TĂNG" and c > 0:
            count += 1
        elif direction == "GIẢM" and c < 0:
            count += 1
        elif direction == "ĐI NGANG" and c == 0:
            count += 1
        else:
            break

    return f"{direction} liên tục {count} ngày"


def _find_support_resistance(rows: List[Dict[str, Any]], lookback: int = 20):
    """Tìm mức hỗ trợ (Low gần nhất) và kháng cự (High gần nhất)."""
    recent = rows[-lookback:] if len(rows) >= lookback else rows
    lows = [r['low'] for r in recent if r['low'] is not None]
    highs = [r['high'] for r in recent if r['high'] is not None]

    support = min(lows) if lows else None
    resistance = max(highs) if highs else None
    return support, resistance


def analyze_price_data(
    days: int = 30,
    start_time: str = None,
    end_time: str = None,
) -> str:
    """
    Phân tích dữ liệu giá dầu và trả về kết quả dạng text.

    Workflow:
    1. Đọc toàn bộ CSV
    2. Lấy tất cả dữ liệu đến end_time (bao gồm trước start_time để tính SMA)
    3. Tính các chỉ báo kỹ thuật
    4. Lọc bảng giá hiển thị theo khoảng [start_time, end_time]

    Args:
        days: Số ngày hiển thị trong bảng giá (mặc định 30)
        start_time: Ngày bắt đầu lọc (override .env)
        end_time: Ngày kết thúc lọc (override .env)

    Returns:
        Kết quả phân tích dạng text có cấu trúc
    """
    all_rows = _load_price_data()
    if not all_rows:
        return "[Lỗi] Không tìm thấy hoặc không thể đọc file dữ liệu giá dầu."

    # Xác định end_time thực tế
    end_str = (end_time or Config.SIMULATION_END_TIME).strip()
    end_dt = _parse_date(end_str) if end_str else None

    # Lấy tất cả rows đến end_time (cần dữ liệu trước start_time để tính SMA)
    if end_dt:
        rows_until_end = [r for r in all_rows if r['date'] <= end_dt]
    else:
        rows_until_end = all_rows

    if not rows_until_end:
        return f"[Lỗi] Không có dữ liệu giá trong khoảng thời gian yêu cầu (đến {end_str})."

    # --- Tính các chỉ báo kỹ thuật ---
    prices = [r['price'] for r in rows_until_end]
    changes = [r['change_pct'] for r in rows_until_end]

    latest = rows_until_end[-1]
    latest_price = latest['price']
    latest_date = latest['date_str']
    latest_change = latest['change_pct']

    sma_5 = _compute_sma(prices, 5)
    sma_10 = _compute_sma(prices, 10)
    sma_20 = _compute_sma(prices, 20)

    # Xu hướng ngắn hạn: giá so với SMA5 và SMA10
    if sma_5 and sma_10:
        if latest_price > sma_5 > sma_10:
            short_trend = "TĂNG (giá > SMA5 > SMA10)"
        elif latest_price < sma_5 < sma_10:
            short_trend = "GIẢM (giá < SMA5 < SMA10)"
        elif latest_price > sma_5:
            short_trend = "TĂNG NHẸ (giá > SMA5, nhưng SMA5 < SMA10)"
        else:
            short_trend = "ĐI NGANG / KHÔNG RÕ XU HƯỚNG"
    else:
        short_trend = "Không đủ dữ liệu"

    # Xu hướng trung hạn: SMA5 so với SMA20
    if sma_5 and sma_20:
        if sma_5 > sma_20:
            mid_trend = "TĂNG (SMA5 > SMA20)"
        elif sma_5 < sma_20:
            mid_trend = "GIẢM (SMA5 < SMA20)"
        else:
            mid_trend = "ĐI NGANG"
    else:
        mid_trend = "Không đủ dữ liệu"

    streak = _compute_streak(changes)
    support, resistance = _find_support_resistance(rows_until_end)

    # Volatility: trung bình |change%| gần đây vs tổng thể
    recent_abs_changes = [abs(c) for c in changes[-10:] if c is not None]
    all_abs_changes = [abs(c) for c in changes if c is not None]
    recent_vol = sum(recent_abs_changes) / len(recent_abs_changes) if recent_abs_changes else 0
    overall_vol = sum(all_abs_changes) / len(all_abs_changes) if all_abs_changes else 0

    if recent_vol > overall_vol * 1.5:
        vol_assessment = f"CAO (gần đây {recent_vol:.2f}% vs TB {overall_vol:.2f}%)"
    elif recent_vol < overall_vol * 0.7:
        vol_assessment = f"THẤP (gần đây {recent_vol:.2f}% vs TB {overall_vol:.2f}%)"
    else:
        vol_assessment = f"BÌNH THƯỜNG (gần đây {recent_vol:.2f}% vs TB {overall_vol:.2f}%)"

    # Biến động trong khoảng simulation
    sim_rows = _filter_by_time_range(rows_until_end, start_time, end_time)
    if sim_rows and len(sim_rows) >= 2:
        sim_start_price = sim_rows[0]['price']
        sim_end_price = sim_rows[-1]['price']
        sim_change = ((sim_end_price - sim_start_price) / sim_start_price) * 100
        sim_high = max(r['high'] for r in sim_rows if r['high'] is not None)
        sim_low = min(r['low'] for r in sim_rows if r['low'] is not None)
    else:
        sim_start_price = sim_end_price = sim_change = sim_high = sim_low = None

    # Ngày biến động mạnh nhất
    if changes:
        valid_rows = [(r, c) for r, c in zip(rows_until_end, changes) if c is not None]
        if valid_rows:
            max_up_row = max(valid_rows, key=lambda x: x[1])
            max_down_row = min(valid_rows, key=lambda x: x[1])
        else:
            max_up_row = max_down_row = None
    else:
        max_up_row = max_down_row = None

    # --- Xây dựng output text ---
    lines = []
    lines.append("═══════════════════════════════════════════════════════════════")
    lines.append("              PHÂN TÍCH DỮ LIỆU GIÁ DẦU (Brent Crude)")
    lines.append("═══════════════════════════════════════════════════════════════")
    lines.append("")

    lines.append("**Giá hiện tại**")
    lines.append(f"  Ngày: {latest_date}")
    lines.append(f"  Giá đóng cửa: ${latest_price:.2f}")
    lines.append(f"  Thay đổi: {latest_change:+.2f}%" if latest_change is not None else "  Thay đổi: N/A")
    lines.append("")

    # Khoảng simulation
    start_label = (start_time or Config.SIMULATION_START_TIME).strip() or "N/A"
    end_label = (end_time or Config.SIMULATION_END_TIME).strip() or "N/A"
    lines.append(f"**Khoảng thời gian mô phỏng**: {start_label} → {end_label}")
    if sim_change is not None:
        lines.append(f"  Giá đầu kỳ: ${sim_start_price:.2f}")
        lines.append(f"  Giá cuối kỳ: ${sim_end_price:.2f}")
        lines.append(f"  Biến động kỳ mô phỏng: {sim_change:+.2f}%")
        lines.append(f"  Đỉnh trong kỳ: ${sim_high:.2f}")
        lines.append(f"  Đáy trong kỳ: ${sim_low:.2f}")
    lines.append("")

    lines.append("**Chỉ báo kỹ thuật**")
    lines.append(f"  SMA 5 ngày:  ${sma_5:.2f}" if sma_5 else "  SMA 5 ngày:  N/A")
    lines.append(f"  SMA 10 ngày: ${sma_10:.2f}" if sma_10 else "  SMA 10 ngày: N/A")
    lines.append(f"  SMA 20 ngày: ${sma_20:.2f}" if sma_20 else "  SMA 20 ngày: N/A")
    lines.append(f"  Xu hướng ngắn hạn: {short_trend}")
    lines.append(f"  Xu hướng trung hạn: {mid_trend}")
    lines.append(f"  Momentum: {streak}")
    lines.append(f"  Biến động (Volatility): {vol_assessment}")
    if support is not None:
        lines.append(f"  Hỗ trợ gần nhất (20 ngày): ${support:.2f}")
    if resistance is not None:
        lines.append(f"  Kháng cự gần nhất (20 ngày): ${resistance:.2f}")
    lines.append("")

    # Sự kiện giá đáng chú ý
    lines.append("**Sự kiện giá đáng chú ý**")
    if max_up_row:
        r, c = max_up_row
        lines.append(f"  Tăng mạnh nhất: {r['date_str']} → {c:+.2f}% (giá ${r['price']:.2f})")
    if max_down_row:
        r, c = max_down_row
        lines.append(f"  Giảm mạnh nhất: {r['date_str']} → {c:+.2f}% (giá ${r['price']:.2f})")
    lines.append("")

    # Bảng giá chi tiết trong khoảng simulation
    display_rows = sim_rows if sim_rows else rows_until_end[-days:]
    lines.append(f"**Bảng giá chi tiết** ({len(display_rows)} ngày giao dịch)")
    lines.append(f"  {'Ngày':<12} {'Giá':>8} {'Mở cửa':>8} {'Cao':>8} {'Thấp':>8} {'KL':>10} {'Thay đổi':>9}")
    lines.append(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*9}")

    for r in display_rows:
        open_s = f"${r['open']:.2f}" if r['open'] is not None else "N/A"
        high_s = f"${r['high']:.2f}" if r['high'] is not None else "N/A"
        low_s = f"${r['low']:.2f}" if r['low'] is not None else "N/A"
        chg_s = f"{r['change_pct']:+.2f}%" if r['change_pct'] is not None else "N/A"
        lines.append(
            f"  {r['date_str']:<12} ${r['price']:>7.2f} {open_s:>8} {high_s:>8} {low_s:>8} {r['volume']:>10} {chg_s:>9}"
        )

    lines.append("")
    lines.append("═══════════════════════════════════════════════════════════════")

    return "\n".join(lines)


def get_price_summary() -> str:
    """
    Tạo bản tóm tắt ngắn gọn về dữ liệu giá dầu, dùng để inject vào
    Planning prompt (PLAN_USER_PROMPT_TEMPLATE).

    Khác với analyze_price_data() (trả về phân tích đầy đủ cho ReACT tool),
    hàm này chỉ trả về thông tin cốt lõi (~15 dòng) để LLM hiểu bối cảnh
    giá khi lên outline, không làm phình context.

    Returns:
        Chuỗi tóm tắt. Trả về "(Không có dữ liệu giá dầu)" nếu không đọc được.
    """
    all_rows = _load_price_data()
    if not all_rows:
        return "(Không có dữ liệu giá dầu)"

    end_str = Config.SIMULATION_END_TIME
    end_dt = _parse_date(end_str) if end_str else None

    if end_dt:
        rows_until_end = [r for r in all_rows if r['date'] <= end_dt]
    else:
        rows_until_end = all_rows

    if not rows_until_end:
        return "(Không có dữ liệu giá trong khoảng thời gian mô phỏng)"

    prices = [r['price'] for r in rows_until_end]
    changes = [r['change_pct'] for r in rows_until_end]
    latest = rows_until_end[-1]

    sma_5 = _compute_sma(prices, 5)
    sma_10 = _compute_sma(prices, 10)
    sma_20 = _compute_sma(prices, 20)

    # Xu hướng ngắn hạn
    if sma_5 and sma_10:
        if latest['price'] > sma_5 > sma_10:
            short_trend = "TĂNG"
        elif latest['price'] < sma_5 < sma_10:
            short_trend = "GIẢM"
        else:
            short_trend = "ĐI NGANG"
    else:
        short_trend = "N/A"

    streak = _compute_streak(changes)
    support, resistance = _find_support_resistance(rows_until_end)

    # Biến động trong khoảng simulation
    sim_rows = _filter_by_time_range(rows_until_end)
    if sim_rows and len(sim_rows) >= 2:
        sim_change = ((sim_rows[-1]['price'] - sim_rows[0]['price']) / sim_rows[0]['price']) * 100
        sim_change_str = f"{sim_change:+.2f}%"
    else:
        sim_change_str = "N/A"

    # 7 ngày gần nhất: đếm số ngày tăng
    recent_7 = [c for c in changes[-7:] if c is not None]
    up_days = sum(1 for c in recent_7 if c > 0)

    lines = [
        f"- Giá hiện tại (ngày {latest['date_str']}): ${latest['price']:.2f}, thay đổi ngày: {latest['change_pct']:+.2f}%" if latest['change_pct'] is not None else f"- Giá hiện tại: ${latest['price']:.2f}",
        f"- Khoảng mô phỏng ({Config.SIMULATION_START_TIME} → {Config.SIMULATION_END_TIME}): biến động {sim_change_str}",
        f"- Xu hướng ngắn hạn: {short_trend}",
        f"- Momentum: {streak}",
        f"- 7 ngày gần nhất: {up_days}/{len(recent_7)} ngày tăng",
    ]

    if sma_5:
        lines.append(f"- SMA5: ${sma_5:.2f}, SMA10: ${sma_10:.2f}" + (f", SMA20: ${sma_20:.2f}" if sma_20 else ""))
    if support is not None and resistance is not None:
        lines.append(f"- Hỗ trợ: ${support:.2f}, Kháng cự: ${resistance:.2f}")

    return "\n".join(lines)
