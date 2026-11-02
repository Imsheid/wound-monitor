# -*- coding: utf-8 -*-
"""
سیستم هوشمند پایش روند ترمیم زخم (نسخه وب - Streamlit)
-------------------------------------------------------
- آپلود عکس زخم از مرورگر
- تشخیص خودکار زخم بر اساس اختلاف رنگ آن با رنگ پوست اطراف (فاصله رنگی در فضای Lab)
- شمارش پیکسل‌های داخل زخم و پیکسل‌های دور تا دور (مرز) زخم
- محاسبه‌ی مساحت (پیکسل و در صورت کالیبراسیون، cm²)
- ثبت نتایج چند عکس و نمایش روند ترمیم

اجرا:
    pip install -r requirements.txt
    streamlit run app.py
"""

from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="پایش ترمیم زخم", page_icon="🩹", layout="wide")

st.markdown(
    """
    <style>
    .stMarkdown, .stAlert, label, .stCaption {direction: rtl; text-align: right;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# توابع پردازش تصویر
# ----------------------------------------------------------------------
def make_demo_image(w=900, h=700):
    """یک تصویر ساختگی (پوست + زخم بیضی‌شکل) برای تست برنامه.
    مساحت واقعی زخم ساختگی ≈ π × 150 × 105 ≈ 49,500 پیکسل است."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.ones((h, w, 3), np.float32) * np.array([150, 175, 225], np.float32)  # BGR پوست
    img *= (0.92 + 0.12 * (xx / w))[..., None]  # نور نامساوی

    cx, cy, a, b = w * 0.5, h * 0.5, 150, 105
    ang = np.deg2rad(20)
    xr = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    yr = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    d = (xr / a) ** 2 + (yr / b) ** 2
    wound = np.clip((1.0 - d) * 12, 0, 1)  # لبه‌ی نرم
    wound_col = np.array([45, 45, 175], np.float32)
    texture = rng.normal(0, 10, (h, w, 1))
    img = img * (1 - wound[..., None]) + (wound_col + texture) * wound[..., None]
    img += rng.normal(0, 5, img.shape)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return np.clip(img, 0, 255).astype(np.uint8)


def segment_wound(roi_bgr, sensitivity=1.0, min_area_pct=0.3, only_largest=True, blur=7):
    """
    زخم را بر اساس اختلاف رنگ با پوست جدا می‌کند.
    مراحل:
      1) تبدیل به فضای Lab (نزدیک به ادراک رنگی انسان)
      2) رنگ پوست = میانه‌ی رنگ در حاشیه‌ی ناحیه‌ی انتخاب‌شده
      3) نقشه‌ی اختلاف رنگ هر پیکسل با رنگ پوست
      4) آستانه‌گذاری Otsu (خودکار) × ضریب حساسیت
      5) پاکسازی نویز و انتخاب ناحیه‌ی زخم
    """
    h, w = roi_bgr.shape[:2]
    blur = blur if blur % 2 == 1 else blur + 1
    smooth = cv2.GaussianBlur(roi_bgr, (blur, blur), 0)

    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0   # L به بازه‌ی 0..100
    lab[..., 1] -= 128.0           # a و b به حالت استاندارد
    lab[..., 2] -= 128.0

    m = max(2, int(min(h, w) * 0.08))
    ring = np.ones((h, w), bool)
    ring[m:h - m, m:w - m] = False
    skin = np.median(lab[ring], axis=0)

    de = np.sqrt(((lab - skin) ** 2).sum(axis=2))
    scale = np.percentile(de, 99.5) + 1e-6
    de8 = np.clip(de / scale * 255, 0, 255).astype(np.uint8)

    otsu, _ = cv2.threshold(de8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu * sensitivity, 1, 254))
    mask = np.where(de8 > thr, 255, 0).astype(np.uint8)

    k = max(3, int(min(h, w) * 0.01) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = h * w * min_area_pct / 100.0
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]

    # ناحیه‌هایی که به لبه‌ی کادر چسبیده‌اند معمولاً پس‌زمینه هستند
    def touches_edge(c):
        x, y, cw, ch = cv2.boundingRect(c)
        return x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1

    inner = [c for c in cnts if not touches_edge(c)]
    if inner:
        cnts = inner
    if not cnts:
        return None, de8, otsu
    if only_largest:
        cnts = [max(cnts, key=cv2.contourArea)]

    final = np.zeros((h, w), np.uint8)
    cv2.drawContours(final, cnts, -1, 255, thickness=cv2.FILLED)
    return final, de8, otsu


def measure(mask, px_per_cm=None):
    """شمارش پیکسل‌های داخل زخم، پیکسل‌های مرز، محیط و مساحت."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    border = np.zeros_like(mask)
    cv2.drawContours(border, cnts, -1, 255, thickness=1)

    res = {
        "area_px": int(cv2.countNonZero(mask)),          # پیکسل‌های داخل زخم
        "border_px": int(cv2.countNonZero(border)),      # پیکسل‌های دور تا دور زخم
        "perimeter_px": round(sum(cv2.arcLength(c, True) for c in cnts), 1),
        "area_cm2": None,
        "perimeter_cm": None,
    }
    if px_per_cm:
        res["area_cm2"] = round(res["area_px"] / px_per_cm ** 2, 3)
        res["perimeter_cm"] = round(res["perimeter_px"] / px_per_cm, 2)
    return res


# ----------------------------------------------------------------------
# رابط کاربری
# ----------------------------------------------------------------------
st.title("🩹 سیستم هوشمند پایش روند ترمیم زخم")
st.caption("عکس زخم را آپلود کنید؛ برنامه زخم را از روی تفاوت رنگ با پوست تشخیص می‌دهد، "
           "پیکسل‌های دور تا دور آن را می‌شمارد و مساحت را محاسبه می‌کند.")

with st.sidebar:
    st.header("⚙️ تنظیمات")
    sens = st.slider("حساسیت تشخیص (کمتر = ناحیه‌ی بزرگ‌تر)", 0.3, 2.0, 1.0, 0.05)
    only_largest = st.checkbox("فقط بزرگ‌ترین ناحیه (پیشنهادی)", value=True)
    min_area_pct = st.slider("حداقل اندازه‌ی ناحیه (٪ از کادر)", 0.05, 5.0, 0.3, 0.05)
    blur_k = st.slider("میزان صاف‌سازی نویز", 3, 25, 7, 2)

    st.subheader("✂️ ناحیه‌ی اطراف زخم")
    st.caption("اگر عکس پس‌زمینه‌ی زیادی دارد، کادر را دور زخم و کمی پوست سالم تنگ کنید.")
    x0, x1 = st.slider("محدوده‌ی افقی (٪)", 0, 100, (0, 100))
    y0, y1 = st.slider("محدوده‌ی عمودی (٪)", 0, 100, (0, 100))

    st.subheader("📏 کالیبراسیون (تبدیل به cm²)")
    calib = st.radio(
        "روش",
        ["بدون کالیبراسیون (فقط پیکسل)", "پیکسل بر سانتی‌متر معلوم است", "عرض واقعی عکس را می‌دانم"],
    )
    cal_value = None
    if calib == "پیکسل بر سانتی‌متر معلوم است":
        cal_value = st.number_input("تعداد پیکسل در هر 1 سانتی‌متر", min_value=0.1, value=40.0, step=1.0)
    elif calib == "عرض واقعی عکس را می‌دانم":
        cal_value = st.number_input("عرض واقعی صحنه‌ی عکس (cm)", min_value=0.1, value=10.0, step=0.5)

uploaded = st.file_uploader("📤 عکس زخم را آپلود کنید", type=["jpg", "jpeg", "png", "bmp", "webp"])
use_demo = st.checkbox("استفاده از تصویر نمونه‌ی ساختگی برای تست", value=uploaded is None)

if uploaded is not None:
    img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    name = uploaded.name
    if img is None:
        st.error("تصویر خوانده نشد. فرمت فایل را بررسی کنید.")
        st.stop()
elif use_demo:
    img = make_demo_image()
    name = "demo"
    st.info("این یک تصویر ساختگی است. مساحت واقعی زخم ساختگی حدود ۴۹٬۵۰۰ پیکسل است؛ "
            "می‌توانید دقت برنامه را با آن بسنجید.")
else:
    st.info("یک عکس آپلود کنید یا گزینه‌ی تصویر نمونه را فعال کنید.")
    st.stop()

H, W = img.shape[:2]
px_per_cm = None
if calib == "پیکسل بر سانتی‌متر معلوم است":
    px_per_cm = float(cal_value)
elif calib == "عرض واقعی عکس را می‌دانم":
    px_per_cm = W / float(cal_value)

px0, px1 = int(W * x0 / 100), int(W * x1 / 100)
py0, py1 = int(H * y0 / 100), int(H * y1 / 100)
if px1 - px0 < 20 or py1 - py0 < 20:
    st.warning("ناحیه‌ی انتخاب‌شده خیلی کوچک است.")
    st.stop()
roi = img[py0:py1, px0:px1]

mask, de8, otsu = segment_wound(roi, sens, min_area_pct, only_largest, blur_k)
if mask is None:
    st.error("زخمی تشخیص داده نشد. حساسیت را کم کنید، حداقل اندازه را کاهش دهید یا کادر را تنظیم کنید.")
    st.image(img, channels="BGR", caption="تصویر ورودی")
    st.stop()

res = measure(mask, px_per_cm)

# ساخت تصویر نتیجه روی کل عکس
full_mask = np.zeros((H, W), np.uint8)
full_mask[py0:py1, px0:px1] = mask
cnts_full, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
overlay = img.copy()
cv2.drawContours(overlay, cnts_full, -1, (0, 255, 255), thickness=cv2.FILLED)
result = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
thick = max(2, int(min(H, W) / 300))
cv2.drawContours(result, cnts_full, -1, (0, 255, 0), thick)
cv2.rectangle(result, (px0, py0), (px1, py1), (255, 120, 0), max(1, thick // 2))

original_marked = img.copy()
cv2.rectangle(original_marked, (px0, py0), (px1, py1), (255, 120, 0), thick)
heat = cv2.applyColorMap(de8, cv2.COLORMAP_JET)

st.subheader("📊 نتایج")
c1, c2, c3, c4 = st.columns(4)
c1.metric("پیکسل‌های داخل زخم", f"{res['area_px']:,}")
c2.metric("پیکسل‌های دور تا دور زخم", f"{res['border_px']:,}")
c3.metric("محیط (پیکسل)", f"{res['perimeter_px']:,}")
c4.metric("مساحت (cm²)", f"{res['area_cm2']}" if res["area_cm2"] is not None else "—")
if res["area_cm2"] is None:
    st.caption("برای مساحت بر حسب cm² از نوار کناری کالیبراسیون را انجام دهید.")

i1, i2, i3 = st.columns(3)
i1.image(original_marked, channels="BGR", caption="تصویر اصلی + کادر انتخابی")
i2.image(heat, channels="BGR", caption="نقشه‌ی اختلاف رنگ با پوست (قرمز = متفاوت‌تر)")
i3.image(result, channels="BGR", caption="زخم تشخیص‌داده‌شده (مرز سبز)")

ok, png = cv2.imencode(".png", result)
if ok:
    st.download_button("⬇️ دانلود تصویر نتیجه", png.tobytes(), file_name="wound_result.png", mime="image/png")

# ----------------------------------------------------------------------
# پایش روند
# ----------------------------------------------------------------------
st.divider()
st.subheader("📈 پایش روند ترمیم")
st.caption("هر عکس را تحلیل کنید و با دکمه‌ی زیر به تاریخچه اضافه کنید. "
           "برای مقایسه‌ی درست، عکس‌ها باید با فاصله، زاویه و نور مشابه (و کالیبراسیون یکسان) گرفته شوند.")

if "history" not in st.session_state:
    st.session_state.history = []

label = st.text_input("برچسب این ثبت (مثلاً روز ۳ یا تاریخ)", value=f"{name} - {datetime.now():%H:%M:%S}")
if st.button("➕ افزودن به تاریخچه"):
    st.session_state.history.append({
        "label": label,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **res,
    })

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    col = "area_cm2" if df["area_cm2"].notna().all() else "area_px"
    st.line_chart(df.set_index("label")[col])

    if len(df) >= 2 and df[col].iloc[0] > 0:
        change = (df[col].iloc[0] - df[col].iloc[-1]) / df[col].iloc[0] * 100
        if change > 0:
            st.success(f"مساحت زخم نسبت به اولین ثبت {change:.1f}٪ کاهش یافته است (روند بهبود).")
        else:
            st.warning(f"مساحت زخم نسبت به اولین ثبت {abs(change):.1f}٪ افزایش یافته است؛ بهتر است توسط پزشک بررسی شود.")

    cc1, cc2 = st.columns(2)
    cc1.download_button("⬇️ دانلود تاریخچه (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="wound_history.csv", mime="text/csv")
    if cc2.button("🗑️ پاک کردن تاریخچه"):
        st.session_state.history = []
        st.rerun()

st.divider()
st.caption("⚠️ این برنامه برای پروژه‌ی آموزشی/پژوهشی و پایش کمکی است و جایگزین ارزیابی پزشک نیست.")
