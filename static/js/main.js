// ── Tab switching ──────────────────────────────────────────
function setTab(btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  const type = btn.dataset.type;
  document.getElementById('loai_bds').value = type;

  // Hiện/ẩn fields nhà đất
  const ndFields  = document.getElementById('nha-dat-fields');
  const rowBal    = document.getElementById('row-balcony');
  if (type === 'nha_dat') {
    ndFields.style.display = 'block';
    rowBal.style.display   = 'none';   // không có hướng ban công cho nhà đất
  } else {
    ndFields.style.display = 'none';
    rowBal.style.display   = 'grid';
  }
  // Reset kết quả
  showIdle();
}

// ── Validate ───────────────────────────────────────────────
function validate() {
  const area     = parseFloat(document.getElementById('area_m2').value);
  const bedrooms = parseInt(document.getElementById('bedrooms_num').value);
  if (!area || area < 10) {
    highlight('area_m2', 'Diện tích phải >= 10 m²'); return false;
  }
  if (!bedrooms || bedrooms < 1) {
    highlight('bedrooms_num', 'Số phòng ngủ phải >= 1'); return false;
  }
  return true;
}

function highlight(id, msg) {
  const el = document.getElementById(id);
  el.style.borderColor = 'var(--red)';
  el.focus();
  setTimeout(() => el.style.borderColor = '', 2000);
  showError(msg);
}

// ── Predict ────────────────────────────────────────────────
async function predict() {
  if (!validate()) return;

  const btn    = document.getElementById('btnPredict');
  const btnTxt = document.getElementById('btnText');
  const loader = document.getElementById('btnLoader');
  btn.disabled = true;
  btnTxt.textContent = 'Đang tính...';
  loader.style.display = 'inline-block';

  const loai = document.getElementById('loai_bds').value;

  const payload = {
    area_m2:      parseFloat(document.getElementById('area_m2').value),
    bedrooms_num: parseInt(document.getElementById('bedrooms_num').value),
    district:     document.getElementById('district').value,
    loai_bds:     loai,
    direction:    document.getElementById('direction').value,
    furniture_std:document.getElementById('furniture_std').value,
    legal_std:    document.getElementById('legal_std').value,
  };

  if (loai === 'nha_dat') {
    payload.floors_num   = parseInt(document.getElementById('floors_num').value)    || 0;
    payload.frontage_m   = parseFloat(document.getElementById('frontage_m').value)  || 0;
    payload.road_width_m = parseFloat(document.getElementById('road_width_m').value)|| 0;
  } else {
    payload.floors_num   = 0;
    payload.frontage_m   = 0;
    payload.road_width_m = 0;
  }

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.success) {
      showResult(data, payload.area_m2);
    } else {
      showError(data.error || 'Lỗi không xác định');
    }
  } catch (err) {
    showError('Không kết nối được server. Hãy chắc chắn Flask đang chạy.');
  } finally {
    btn.disabled = false;
    btnTxt.textContent = 'Dự Đoán Giá Trị';
    loader.style.display = 'none';
  }
}

// ── Show result ────────────────────────────────────────────
function showResult(data, area) {
  const price = data.price_billion;

  // Ước tính dao động ±15%
  const low  = (price * 0.85).toFixed(2);
  const high = (price * 1.15).toFixed(2);

  document.getElementById('priceValue').textContent  = data.price_display;
  document.getElementById('priceDetail').textContent =
    `≈ ${formatMillion(price * 1000)} triệu VNĐ`;
  document.getElementById('pricePerM2').textContent  =
    `Giá/m²: ${data.price_per_m2} triệu/m²`;
  document.getElementById('rangeLow').textContent    = low + ' tỷ';
  document.getElementById('rangeHigh').textContent   = high + ' tỷ';
  document.getElementById('modelName').textContent   = data.model_used;
  document.getElementById('r2Score').textContent     = (data.r2_score * 100).toFixed(1) + '%';

  document.getElementById('resultIdle').style.display = 'none';
  document.getElementById('resultError').style.display = 'none';
  document.getElementById('resultData').style.display  = 'block';

  // Animate price count-up
  animateNumber('priceValue', 0, price, 800, v => v.toFixed(2));
}

function showError(msg) {
  document.getElementById('resultIdle').style.display  = 'none';
  document.getElementById('resultData').style.display  = 'none';
  document.getElementById('resultError').style.display = 'block';
  document.getElementById('errorText').textContent     = msg;
}

function showIdle() {
  document.getElementById('resultIdle').style.display  = 'flex';
  document.getElementById('resultData').style.display  = 'none';
  document.getElementById('resultError').style.display = 'none';
}

// ── Helpers ────────────────────────────────────────────────
function formatMillion(val) {
  return val >= 1000
    ? (val / 1000).toFixed(2) + ' tỷ'
    : Math.round(val) + ' triệu';
}

function animateNumber(id, from, to, duration, fmt) {
  const el   = document.getElementById(id);
  const start = performance.now();
  function step(now) {
    const p = Math.min((now - start) / duration, 1);
    const e = p < .5 ? 2*p*p : -1+(4-2*p)*p; // ease in-out
    el.textContent = fmt(from + (to - from) * e);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Reset fields on input ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  ['area_m2','bedrooms_num'].forEach(id => {
    document.getElementById(id).addEventListener('input', showIdle);
  });
  ['district','direction','furniture_std','legal_std',
   'floors_num','frontage_m','road_width_m'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', showIdle);
  });

  // Enter key submit
  document.getElementById('predictForm').addEventListener('keydown', e => {
    if (e.key === 'Enter') predict();
  });
});
