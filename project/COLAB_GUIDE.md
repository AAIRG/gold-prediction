# راهنمای گام‌به‌گام اجرای آزمایش‌ها روی Google Colab (رایگان)

هدف: اجرای آزمایش‌های *Conditioned Normalizing Flows* با بودجهٔ کاهش‌یافته و ثابت
روی ۳ اکانت Colab رایگان، تا همهٔ خروجی‌های لازم برای گزارش نتیجهٔ مشاهده‌شده
(جدول‌ها، شکل‌ها و چک‌پوینت‌ها) آماده باشد. این اجرا ادعای بازتولید آماری کامل
جدول مقاله را ندارد.

> این راهنما فرض می‌کند کد داخل پوشهٔ `project/` است و **هیچ تغییری در کد لازم نیست**.
> مسیر اصلی، سه نوت‌بوک مستقل `notebooks/colab_account{1,2,3}.ipynb` است.

---

## ★ مسیر پیشنهادی (خودکار): همین را انجام بده

1. روی اکانت ۱، `colab_account1.ipynb` را اجرا کن: Exchange و Solar.
2. روی اکانت ۲، `colab_account2.ipynb` را اجرا کن: Electricity و Traffic.
3. روی اکانت ۳، `colab_account3.ipynb` را اجرا کن: Taxi و Wikipedia.
4. هر نوت‌بوک path مستقل `outputs_v2/accountN` و `checkpoints_v2/accountN` دارد،
   هر پنج دقیقه sync می‌کند و پس از قطع‌شدن از آخرین epoch دقیق ادامه می‌دهد.
5. نتیجهٔ هر اجرای کامل، بدون الزام تطابق رتبه‌بندی با مقاله، همان‌طور که مشاهده
   شده است گزارش می‌شود. epoch و seedهای زیر پس از مشاهدهٔ نتیجه تغییر نمی‌کنند.

---

## ۰) قانون طلایی: چه چیزی ثابت است و چه چیزی قابل‌تغییر

طبق محدودیت شما، **طراحی آزمایش‌ها دست‌نخورده می‌ماند** و فقط ابرپارامترهای بهینه‌سازی
را می‌توانید تغییر دهید.

**ثابت (اجازهٔ تغییر ندارید):**
- هر ۶ دیتاست: Exchange، Solar، Electricity، Traffic، Taxi، Wikipedia.
- هر ۳ واریانت مقاله برای هر دیتاست: **LSTM-Real-NVP**، **LSTM-MAF**، **Transformer-MAF**.
- بیس‌لاین **Vec-LSTM** برای هر دیتاست.
- پروتکل ارزیابی: `num_samples: 100`، `num_test_windows` طبق کانفیگ (برابر تعداد
  rolling evaluation هر دیتاست)، dequantize برای Taxi/Wikipedia.
- ساختار مدل: K=5 بلوک، hidden فلو=100، ELU، BatchNorm، لگ‌ها و کوواریت‌های
  Appendix D.1 — همه ثابت.

**بودجهٔ ثابت اجرای فعلی:** `epochs` و تعداد `seed` همان مقادیر ثبت‌شده در سه
نوت‌بوک هستند و برای هر سه flow یک دیتاست یکسان‌اند. `lr`، `batch_size`، scheduler،
داده و metric نیز ثابت می‌مانند.

**تنها مصالحهٔ اجتناب‌ناپذیر — تعداد seed:** مقاله برای CRPS_sum میانگین ۲۰ ران
(و برای CRPS سه ران) گزارش می‌کند. ۲۰ ران × ۳۰ پیکربندی در ۱ روزِ Colab رایگان
شدنی نیست. این «تعداد ران» یک انتخاب آماری است، نه خودِ آزمایش؛ بیشترین چیزی که وقت
اجازه دهد اجرا کنید (توصیه: **۳ seed** برای دیتاست‌های کوچک، حداقل **۱ seed** برای
بزرگ‌ها) و تعداد ران را در گزارش شفاف بنویسید.

---

## ۱) اصولی که باید بدانید

1. **ماندگاری روی Drive حیاتی است.** Colab رایگان با ~۹۰ دقیقه بی‌کاری یا حداکثر
   ~۱۲ ساعت قطع می‌شود. چون `checkpoints_v2/accountN`، `outputs_v2/accountN` و کش دیتاست
   (`data_raw/gluonts`) روی Drive‌اند، بعد از هر قطع‌شدن کافی است همان سلول آموزش را
   دوباره اجرا کنید؛ تِرینر از آخرین epoch ادامه می‌دهد (`checkpoints/<run>.pt`).
2. فایل `_best.pt` در contract فعلی همان **final epoch** است؛ validation selection نداریم.
3. **نام‌گذاری خودکار اجراها:** `<dataset>_<backbone>_<flow>[_seed<seed>]`؛ بیس‌لاین:
   `<dataset>_veclstm`.
4. **GPU:** `Runtime → Change runtime type → T4 GPU`. اسکریپت‌ها خودشان cuda را
   می‌گیرند.
5. **هر اکانت = یک نشست GPU** → موازی‌سازی بین اکانت‌ها.

### هزینهٔ واقعی محاسبات (چرا این طرح شدنی است)
- بیشتر دیتاست‌ها T کوتاهی دارند (Wikipedia فقط ۷۹۲ گام، Taxi ۱۴۸۸)، پس آموزش با
  epoch کمتر سریع است.
- گلوگاه واقعی: **نمونه‌گیری MAF برای D بزرگ** (inverse ترتیبی روی D). ولی چون
  num_test_windows و horizon کوچک‌اند، حتی Wikipedia (D=2000) هر ارزیابی MAF حدود
  ۱۰–۱۵ دقیقه است، نه چند ساعت. پس با تقسیم درست بین اکانت‌ها همه‌چیز در ۱ روز جا
  می‌شود.

---

## ۲) آماده‌سازی یک‌باره (برای هر اکانت)

### ۲.۱ زیپ و آپلود
پوشهٔ `project` را زیپ کنید (`project.zip`) و در Drive هر اکانت در مسیر بدون‌فاصله
`MyDrive/cnf/` بگذارید.

### ۲.۲ سلول اول (فقط بار اول): mount + باز کردن zip
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/cnf', exist_ok=True)
%cd /content/drive/MyDrive/cnf
!test -d project || unzip -q project.zip -d .
%cd /content/drive/MyDrive/cnf/project
!ls
```

### ۲.۳ نصب وابستگی‌ها (هر VM تازه)
```python
%pip -q install "gluonts==0.17.0" orjson
!nvidia-smi -L
```

### ۲.۴ سلول شروع هر جلسه (بعد از هر reconnect)
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/cnf/project
import os; print('CWD:', os.getcwd())
```

---

## ۳) اهرم‌های مجاز و ساخت واریانت‌ها (کاری که نوت‌بوک خودکار می‌کند)

اگر از نوت‌بوک استفاده می‌کنی این بخش خودکار انجام می‌شود؛ فقط برای شفافیت آمده. تنها
برای هر دیتاست هر ۳ واریانت ساخته می‌شود و runner بودجهٔ ثابت همان نوت‌بوک را روی
همهٔ آن‌ها اعمال می‌کند.

مقادیرِ دقیقی که نوت‌بوک استفاده می‌کند (درون هر دیتاست برای هر ۳ واریانت **برابر** →
مقایسه منصفانه؛ بین دیتاست‌های مختلف می‌تواند فرق کند چون هر دیتاست مقایسهٔ مستقل است):

```python
import pathlib, re

# اهرم‌های مجاز — دقیقاً مثل نوت‌بوک:
EPOCHS = {'exchange': 25, 'solar': 25, 'electricity': 15,
          'traffic': 20, 'taxi': 20, 'wikipedia': 15}
LR    = '1.0e-3'     # نرخ یادگیری مقاله
BATCH = {}           # OOM؟ مثلا {'wikipedia': 8} → روی هر ۳ واریانت همان دیتاست اعمال می‌شود

def _set(txt, key, val):
    return re.sub(rf'(?m)^  {key}: .*$', f'  {key}: {val}', txt)

def make_variant(ds, name, backbone, flow):
    txt = pathlib.Path(f'configs/{ds}.yaml').read_text(encoding='utf-8')
    txt = re.sub(r'(?m)^  type: (realnvp|maf).*$',      f'  type: {flow}', txt)
    txt = re.sub(r'(?m)^  type: (lstm|transformer).*$', f'  type: {backbone}', txt)
    txt = _set(txt, 'epochs', EPOCHS[ds]); txt = _set(txt, 'lr', LR)
    if ds in BATCH:
        txt = _set(txt, 'batch_size', BATCH[ds])
    pathlib.Path(f'configs/{ds}__{name}.yaml').write_text(txt, encoding='utf-8')

VARIANTS = {'lstm_realnvp': ('lstm', 'realnvp'),
            'lstm_maf': ('lstm', 'maf'),
            'transformer_maf': ('transformer', 'maf')}

for ds in ['exchange', 'solar', 'electricity', 'traffic', 'taxi', 'wikipedia']:
    for name, (bk, fl) in VARIANTS.items():
        make_variant(ds, name, bk, fl)
```

- نام اجرا از کانفیگ ساخته می‌شود؛ `configs/exchange__lstm_maf.yaml` → اجرای
  `exchange_lstm_maf`.
- **انصاف:** چون هر ۳ واریانتِ یک دیتاست از یک کانفیگ پایهٔ مشترک و همان `EPOCHS[ds]`
  ساخته می‌شوند، `epochs`/`lr`/`batch`/`seed` آن‌ها برابر است. اگر برای OOM بَچ را کم
  کردی، `BATCH[ds]` را بگذار تا **برای هر ۳ واریانت** یکسان اعمال شود.

---

## ۴) تقسیم کار بین اکانت‌ها (پوششِ کاملِ Table 1)

هر دیتاست باید هر ۳ واریانت + Vec-LSTM + random را بگیرد. «heavy» یعنی ارزیابی MAF
روی D بزرگ (کندترین بخش) که بین اکانت‌ها پخش شده است.

| اکانت | دیتاست‌ها | واریانت‌ها (هر کدام) | seed | heavy MAF eval |
|---|---|---|---|---|
| ۱ | Exchange، Solar | RealNVP, LSTM-MAF, Transformer-MAF, Vec-LSTM, random | Exchange: 1,2,3 ؛ Solar: 1 (+2,3 اگر شد) | کم (D≤۱۳۷) |
| ۲ | Electricity، Traffic | همان ۵ | 1 (+2,3 اگر شد) | ۲ (traffic MAF×۲) |
| ۳ | Taxi، Wikipedia | همان ۵ | 1 (+2,3 اگر شد) | ۴ (taxi+wiki MAF×۲) |

> اگر اکانت ۳ عقب افتاد، Wikipedia را به اکانت ۱ منتقل کنید (اکانت ۱ بعد از Exchange
> سبک است). هدف: هیچ دیتاست/واریانتی حذف نشود.

اکانت ۱ علاوه بر این‌ها مسئول **تست‌ها + overfit + آزمایش لوله‌ها** هم هست.

---

## ۵) گام‌به‌گام اجرا (معادل دستیِ سوییپ نوت‌بوک)

> اگر نوت‌بوک را اجرا می‌کنی، این بخش را رد کن. این‌ها همان کاری‌اند که سلولِ سوییپ
> خودکار انجام می‌دهد، برای وقتی که ترجیح می‌دهی خط‌فرمان بزنی.

### گام A — صحت‌سنجی (اکانت ۱)
```python
!python -m pytest tests -q
!python scripts/train.py --config configs/exchange.yaml --overfit-one-window --epochs 15 --batches-per-epoch 30 --no-resume
```

### گام B — الگوی اجرای یک دیتاست کامل
برای هر دیتاست `<ds>`، این بلوک همهٔ واریانت‌ها و بیس‌لاین‌ها را اجرا می‌کند.
`SEEDS` را بسته به سنگینی دیتاست و وقت تنظیم کنید (`"1 2 3"` یا `"1"`).

```python
ds = 'exchange'          # یا solar / electricity / traffic / taxi / wikipedia
SEEDS = '1 2 3'          # دیتاست بزرگ: '1'

for name in ['lstm_realnvp', 'lstm_maf', 'transformer_maf']:
    !python scripts/run_experiments.py --config configs/{ds}__{name}.yaml --seeds {SEEDS}

# بیس‌لاین Vec-LSTM (بک‌بون LSTM؛ از کانفیگ lstm_realnvp استفاده می‌کنیم تا حتماً LSTM باشد)
!python scripts/run_experiments.py --config configs/{ds}__lstm_realnvp.yaml --seeds {SEEDS} --baseline veclstm

# بیس‌لاین تصادفی (سلامت متریک، بدون آموزش)
!python scripts/evaluate.py --config configs/{ds}__lstm_realnvp.yaml --baseline random --run-name {ds}_random
```

### گام C — شکل‌ها
- **fan chartها خودکار ساخته می‌شوند** توسط `run_experiments` (برای هر واریانت و seed؛
  نام: `outputs/fan_chart_<ds>_<variant>_seed1_series0.png`). کاری لازم نیست.
- برای **Figure 5 (کوواریانس متقاطع)** فقط یک ارزیابیِ اضافه با واریانت سریع RealNVP
  کافی است (برای **Wikipedia** به‌خاطر مصرف RAM آن را رد کن):
```python
ds = 'exchange'   # برای هر دیتاست به‌جز wikipedia
!python scripts/evaluate.py --config configs/{ds}__lstm_realnvp.yaml \
    --checkpoint checkpoints/{ds}_lstm_realnvp_seed1_best.pt \
    --run-name {ds}_lstm_realnvp_fig --cross-covariance
```
> نوت‌بوک دقیقاً همین کار را می‌کند: fan chart از خودِ `run_experiments`، و
> cross-covariance فقط از RealNVP و با رد کردن Wikipedia.

### گام D — آزمایش لوله‌ها (§5.1، شکل‌های ۳ و ۴) — اکانت ۱
```python
!python scripts/pipes_experiment.py --timesteps 10000 --epochs 20 --flow maf
!python scripts/pipes_experiment.py --timesteps 10000 --epochs 20 --flow realnvp
```

### گام E — بستن هر اکانت
```python
!cat outputs/results_table.md
!zip -qr /content/drive/MyDrive/cnf/outputs_account1.zip outputs
```

---

## ۶) دستورهای آمادهٔ هر اکانت (معادل دستیِ سوییپ نوت‌بوک)

> این‌ها همان چیزی‌اند که سلولِ سوییپ نوت‌بوک با انتخاب `ACCOUNT` اجرا می‌کند. اگر
> نوت‌بوک را می‌زنی لازم نیست این‌ها را دستی وارد کنی.

**اکانت ۱** (بعد از سلول‌های راه‌اندازی + سلول ساخت واریانت‌ها + گام A):
```python
for ds, seeds in [('exchange', '1 2 3'), ('solar', '1')]:
    for name in ['lstm_realnvp', 'lstm_maf', 'transformer_maf']:
        !python scripts/run_experiments.py --config configs/{ds}__{name}.yaml --seeds {seeds}
    !python scripts/run_experiments.py --config configs/{ds}__lstm_realnvp.yaml --seeds {seeds} --baseline veclstm
    !python scripts/evaluate.py --config configs/{ds}__lstm_realnvp.yaml --baseline random --run-name {ds}_random
# سپس گام C برای exchange و solar، و گام D (لوله‌ها)
```

**اکانت ۲:**
```python
for ds, seeds in [('electricity', '1'), ('traffic', '1')]:
    for name in ['lstm_realnvp', 'lstm_maf', 'transformer_maf']:
        !python scripts/run_experiments.py --config configs/{ds}__{name}.yaml --seeds {seeds}
    !python scripts/run_experiments.py --config configs/{ds}__lstm_realnvp.yaml --seeds {seeds} --baseline veclstm
    !python scripts/evaluate.py --config configs/{ds}__lstm_realnvp.yaml --baseline random --run-name {ds}_random
```

**اکانت ۳:**
```python
for ds, seeds in [('taxi', '1'), ('wikipedia', '1')]:
    for name in ['lstm_realnvp', 'lstm_maf', 'transformer_maf']:
        !python scripts/run_experiments.py --config configs/{ds}__{name}.yaml --seeds {seeds}
    !python scripts/run_experiments.py --config configs/{ds}__lstm_realnvp.yaml --seeds {seeds} --baseline veclstm
    !python scripts/evaluate.py --config configs/{ds}__lstm_realnvp.yaml --baseline random --run-name {ds}_random
```

> اگر اکانتی زود تمام کرد: **seedهای بیشتر** برای همان دیتاست‌ها اضافه کند
> (`--seeds 2 3`) تا mean±SE قوی‌تر شود — از کوچک‌ترین دیتاست شروع کنید. این کار
> «افزودن ران» است، نه تغییر آزمایش.

---

## ۷) مدیریت قطع‌شدن (auto-resume)

- بعد از قطع: سلول شروع جلسه (۲.۴) را اجرا کنید، بعد **همان دستور را دوباره بزنید**؛
  پیام `Resuming from ...` می‌آید.
- برای فقط ارزیابیِ دوباره بدون آموزش:
  `!python scripts/run_experiments.py --config configs/<ds>__<name>.yaml --seeds 1 --skip-training`
- تب Colab را فعال نگه دارید تا زودتر قطع نشود.

---

## ۸) اهرم‌های مجاز اگر وقت کم آمد (فقط این‌ها)

| اهرم | چطور | یادداشت گزارش |
|---|---|---|
| کم‌کردن epoch | `EPOCHS[ds]` (نوت‌بوک) یا `--epochs N` | «به‌جای ۴۰، N ایپاک» |
| افزایش batch_size | `BATCH[ds]` (نوت‌بوک) یا ویرایش `  batch_size:` در کانفیگ | سریع‌تر؛ مراقب OOM |
| افزایش lr | `LR` (مثلاً `1.5e-3`) | جبران epoch کمتر |
| تعداد seed | `--seeds` | تنها مصالحهٔ آماری؛ عدد را بنویسید |

> **مجاز نیست:** حذف دیتاست/واریانت، کم‌کردن `num_samples` از ۱۰۰، تغییر
> `num_test_windows`، یا جایگزینی MAF با RealNVP. این‌ها خودِ آزمایش‌اند.

اگر یک ارزیابی MAF روی D بزرگ خیلی طول کشید، **راهش epoch کمتر برای آموزش است، نه
کم‌کردن نمونه‌ها**؛ نمونه‌گیری با ۱۰۰ نمونه باید کامل اجرا شود.

---

## ۹) جمع‌آوری و ادغام نتایج

۱. سه فایل `outputs_account{1,2,3}_v2.zip` را از Drive هر اکانت دانلود و در
   پوشه‌های `outputs_all_v2/acc1|acc2|acc3` باز کنید.
۲. جدول نهایی:
```python
import glob, pandas as pd
frames = [pd.read_csv(f) for f in glob.glob('outputs_all_v2/*/results.csv')]
df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset=['run'], keep='last')
df.to_csv('outputs_all_v2/results_merged.csv', index=False)

# CRPS_sum برحسب مدل × دیتاست (میانگین روی seedها)
print(df.pivot_table(index='model', columns='dataset',
                     values='crps_sum', aggfunc='mean').round(4))
```
۳. فایل‌های `aggregate_*.json` ضروری نیستند. اگر mean±SE برای گزارش لازم است، آن
   را پس از merge مستقیماً از `results_merged.csv` و روی seedهای هر مدل محاسبه کنید.

---

## ۱۰) چک‌لیست خروجی‌های لازم برای گزارش

- [ ] لاگ سبز `pytest` + لاگ `--overfit-one-window`.
- [ ] `results_merged.csv` با CRPS_sum / CRPS / MSE برای **هر ۶ دیتاست × ۳ واریانت**
      + Vec-LSTM؛ mean±SE در صورت نیاز از همین CSV محاسبه شود.
- [ ] برای هر دیتاست×واریانت: `outputs/loss_<run>_seed1.png`.
- [ ] fan chartها (`fan_chart_*_seed1_series0.png`، خودکار از `run_experiments`) و
      `cross_covariance_*_fig.png` (Figure 5؛ برای همه به‌جز Wikipedia).
- [ ] آزمایش لوله‌ها: `pipes_series.png`, `pipes_lstm_maf_covariance.png`,
      `pipes_lstm_maf_cross_covariance.png`, `pipes_report.json`.
- [ ] مقایسهٔ اعداد با ستون‌های Table 1 مقاله (عدد `crps_sum` ما نرمالایزشده و
      قابل‌مقایسه است).
- [ ] یادداشت شفاف در گزارش: epoch/lr/batch استفاده‌شده و تعداد seed.

---

## ۱۱) برنامهٔ ۱ روزه (سه اکانت موازی)

| زمان | اکانت ۱ | اکانت ۲ | اکانت ۳ |
|---|---|---|---|
| 0:00–0:30 | راه‌اندازی + انتخاب `ACCOUNT=1` | راه‌اندازی + `ACCOUNT=2` | راه‌اندازی + `ACCOUNT=3` |
| 0:30–1:00 | pytest + overfit | شروع Electricity | شروع Taxi |
| 1:00–3:30 | Exchange (۳ seed) کامل | Electricity کامل + شروع Traffic | Taxi کامل + شروع Wikipedia |
| 3:30–5:30 | Solar کامل + لوله‌ها | Traffic کامل (۲ eval سنگین) | Wikipedia کامل (۲ eval سنگین) |
| 5:30–6:30 | شکل‌ها + zip | شکل‌ها + zip | شکل‌ها + zip |
| 6:30–8:00 | دانلود ۳ zip → ادغام → ساخت جدول‌ها/شکل‌ها | | |
| 8:00+ | seedهای اضافه اگر وقت ماند، سپس شروع گزارش | | |

---

## ۱۲) عیب‌یابی

- **`ModuleNotFoundError: gluonts`** → سلول نصب (۲.۳).
- **`No module named flowforecast`** → CWD باید `.../cnf/project` باشد (سلول ۲.۴).
- **`CUDA out of memory`** (Wiki/Traffic/Taxi) → `batch_size` را در همان کانفیگ کم
  کنید (مجاز). مثلاً Wikipedia را 8 یا 4.
- **دانلود دیتاست تکرار می‌شود** → یعنی از داخل پوشهٔ Drive اجرا نمی‌کنید؛ کش
  `data_raw/gluonts` باید روی Drive بماند.
- **ارزیابی MAF روی D بزرگ کند است** → طبیعی است (inverse ترتیبی، §2.1)؛ صبر کنید،
  نمونه‌ها را کم نکنید. برای صرفه‌جویی فقط epoch آموزش را کم کنید.
- **هشدار `WARNING ... training range ...`** → نسخهٔ محلی دیتاست کمی فرق دارد؛ اجرا
  ادامه می‌یابد.
- **`cond_size` مچ نیست هنگام ارزیابی** → همان فایل کانفیگی را بدهید که با آن آموزش
  دادید (`configs/<ds>__<name>.yaml`).

---

## ۱۳) برگهٔ تقلب

```bash
# صحت‌سنجی
python -m pytest tests -q
python scripts/train.py --config configs/exchange.yaml --overfit-one-window --epochs 15 --batches-per-epoch 30 --no-resume

# یک واریانت کامل (mean±SE) — همهٔ واریانت‌ها باید اجرا شوند
python scripts/run_experiments.py --config configs/<ds>__lstm_realnvp.yaml    --seeds 1 2 3
python scripts/run_experiments.py --config configs/<ds>__lstm_maf.yaml        --seeds 1 2 3
python scripts/run_experiments.py --config configs/<ds>__transformer_maf.yaml --seeds 1 2 3
python scripts/run_experiments.py --config configs/<ds>__lstm_realnvp.yaml    --seeds 1 2 3 --baseline veclstm
python scripts/evaluate.py        --config configs/<ds>__lstm_realnvp.yaml    --baseline random --run-name <ds>_random

# شکل‌ها از چک‌پوینت seed=1
python scripts/evaluate.py --config configs/<ds>__lstm_maf.yaml \
    --checkpoint checkpoints/<ds>_lstm_maf_seed1_best.pt --run-name <ds>_lstm_maf_fig --cross-covariance

# آزمایش لوله‌ها
python scripts/pipes_experiment.py --timesteps 10000 --epochs 20 --flow maf
```

اهرم‌های مجاز: `--epochs` (CLI)، و `lr`/`batch_size` از طریق تابع `make_variant`
یا ویرایش خطوط `  lr:` و `  batch_size:` در کانفیگ. بقیهٔ آزمایش دست‌نخورده می‌ماند.
