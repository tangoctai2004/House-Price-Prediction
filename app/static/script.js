document.addEventListener('DOMContentLoaded', () => {
    const typeRadios = document.querySelectorAll('input[name="property_type_toggle"]');
    const chungCuFields = document.querySelectorAll('.chung-cu-only');
    const nhaDatFields = document.querySelectorAll('.nha-dat-only');
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const predictedPriceEl = document.getElementById('predicted-price');
    const resultMessageEl = document.getElementById('result-message');
    const errorEl = document.getElementById('error-message');
    const submitBtn = document.getElementById('submit-btn');
    if (!form || !submitBtn) {
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a');
            if (link && link.href && link.target !== '_blank' && !link.href.startsWith('#') && !link.href.startsWith('javascript:')) {
                e.preventDefault();
                document.body.classList.add('page-transitioning');
                setTimeout(() => {
                    window.location.href = link.href;
                }, 250);
            }
        });
        return;
    }
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    const resetBtn = document.getElementById('reset-btn');

    // Toggle fields based on property type
    function updateFields() {
        const isNhaDat = document.getElementById('type-nha-dat').checked;
        
        if (isNhaDat) {
            chungCuFields.forEach(el => {
                el.style.display = 'none';
            });
            nhaDatFields.forEach(el => {
                el.style.display = 'block';
            });
        } else {
            chungCuFields.forEach(el => {
                el.style.display = 'block';
            });
            nhaDatFields.forEach(el => {
                el.style.display = 'none';
            });
        }
    }

    typeRadios.forEach(radio => {
        radio.addEventListener('change', updateFields);
    });

    // Initial setup
    updateFields();

    // Format currency
    function formatCurrency(value) {
        if (value >= 1000000000) {
            return (value / 1000000000).toFixed(2) + ' Tỷ VNĐ';
        } else {
            return (value / 1000000).toFixed(0) + ' Triệu VNĐ';
        }
    }

    // Animate number count up
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // easeOutQuart
            const easeProgress = 1 - Math.pow(1 - progress, 4);
            const currentVal = Math.floor(easeProgress * (end - start) + start);
            obj.innerHTML = formatCurrency(currentVal);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = formatCurrency(end);
            }
        };
        window.requestAnimationFrame(step);
    }

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        errorEl.textContent = '';
        
        // UI Loading state
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        data.property_type = document.getElementById('type-nha-dat').checked ? 'nha_dat' : 'chung_cu';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                // Hide form, show result
                form.style.display = 'none';
                resultContainer.style.display = 'block';
                
                // Animate price
                const finalPrice = result.predicted_price_vnd;
                animateValue(predictedPriceEl, 0, finalPrice, 2000);
                
                resultMessageEl.textContent = result.message;

                // Render similar properties
                const similarContainer = document.getElementById('similar-properties-container');
                const similarList = document.getElementById('similar-list');
                similarList.innerHTML = ''; // Clear old

                if (result.similar_properties && result.similar_properties.length > 0) {
                    similarContainer.style.display = 'block';
                    result.similar_properties.forEach(prop => {
                        const card = document.createElement('a');
                        const type = prop.property_type === 'nha_dat' ? 'nha_dat' : 'chung_cu';
                        const id = Number.isInteger(Number(prop.id)) ? Number(prop.id) : 0;
                        const image = safeImageUrl(prop.image);
                        const title = escapeHtml(prop.title || prop.district || '');
                        const district = escapeHtml(prop.district || '');
                        const price = Number.isFinite(Number(prop.price_billion)) ? Number(prop.price_billion) : 0;
                        const area = Number.isFinite(Number(prop.area_m2)) ? Number(prop.area_m2) : 0;
                        const desc = escapeHtml(prop.desc || '');
                        card.href = `/property/${type}/${id}`;
                        card.className = 'similar-card';

                        const imgHtml = image
                            ? `<img class="card-thumbnail" src="${image}" alt="${district}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="card-thumbnail-placeholder" style="display:none">🏠</div>`
                            : `<div class="card-thumbnail-placeholder">🏠</div>`;

                        card.innerHTML = `
                            ${imgHtml}
                            <div class="card-body">
                                <p class="card-title">${title}</p>
                                <p class="card-price">${price} Tỷ VNĐ</p>
                                <div class="card-meta">
                                    <span>📐 ${area} m²</span>
                                    <span>📍 ${district}</span>
                                </div>
                                <p class="card-meta">${desc}</p>
                            </div>
                        `;
                        similarList.appendChild(card);
                    });
                } else {
                    similarContainer.style.display = 'none';
                }
            } else {
                errorEl.textContent = result.error || 'Có lỗi xảy ra khi dự đoán.';
            }
        } catch (error) {
            errorEl.textContent = 'Không thể kết nối tới server.';
            console.error(error);
        } finally {
            // Restore UI state
            btnText.style.display = 'block';
            spinner.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        resultContainer.style.display = 'none';
        document.getElementById('similar-properties-container').style.display = 'none';
        form.style.display = 'block';
        form.reset();
        errorEl.textContent = '';
        updateFields();
    });

    // Smooth page transitions
    document.addEventListener('click', (e) => {
        const link = e.target.closest('a');
        if (link && link.href && link.target !== '_blank' && !link.href.startsWith('#') && !link.href.startsWith('javascript:')) {
            e.preventDefault();
            document.body.classList.add('page-transitioning');
            setTimeout(() => {
                window.location.href = link.href;
            }, 250);
        }
    });
});

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, ch => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[ch]));
}

function safeImageUrl(value) {
    const url = String(value || '').trim();
    return /^https?:\/\//i.test(url) ? escapeHtml(url) : '';
}

/* ==========================================
   AUTH & MODAL FUNCTIONS
   ========================================== */

function openAuthModal(mode) {
    const existing = document.getElementById('auth-modal');
    if (existing) existing.remove();

    const isLogin = mode === 'login';
    const modal = document.createElement('div');
    modal.id = 'auth-modal';
    modal.className = 'auth-modal-overlay';
    modal.innerHTML = `
        <div class="auth-modal-box" onclick="event.stopPropagation()">
            <button class="auth-modal-close" onclick="closeAuthModal()">✕</button>
            <div class="auth-modal-title">${isLogin ? 'Đăng nhập' : 'Tạo tài khoản'}</div>
            <p class="auth-modal-sub">${isLogin ? 'Đăng nhập để truy cập tài khoản của bạn' : 'Đăng ký tài khoản mới miễn phí'}</p>
            <a href="/auth/google" class="btn-google">
                <svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.99 7.28-2.66l-3.57-2.77c-.99.66-2.25 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
                <span>Tiếp tục với Google</span>
            </a>
            <div class="auth-divider">hoặc</div>
            ${!isLogin ? '<div class="auth-field"><label>Họ và tên</label><input type="text" id="auth-name" placeholder="Nguyễn Văn A"></div>' : ''}
            <div class="auth-field"><label>Email</label><input type="email" id="auth-email" placeholder="email@example.com"></div>
            <div class="auth-field"><label>Mật khẩu</label><input type="password" id="auth-pass" placeholder="••••••••"></div>
            <div id="auth-error" class="auth-modal-error"></div>
            <button class="auth-submit" id="auth-submit-btn" onclick="doAuth('${mode}')">${isLogin ? 'Đăng nhập →' : 'Tạo tài khoản →'}</button>
            <div class="auth-switch">${isLogin ? 'Chưa có tài khoản? <a onclick="openAuthModal(\'register\')">Đăng ký</a>' : 'Đã có tài khoản? <a onclick="openAuthModal(\'login\')">Đăng nhập</a>'}</div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => { if (e.target === modal) closeAuthModal(); });
    modal.querySelectorAll('input').forEach(inp => inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') doAuth(mode); }));
    requestAnimationFrame(() => { modal.classList.add('active'); });
}

function closeAuthModal() {
    const modal = document.getElementById('auth-modal');
    if (!modal) return;
    modal.classList.remove('active');
    setTimeout(() => modal.remove(), 300);
}

async function doAuth(mode) {
    const email = document.getElementById('auth-email').value.trim();
    const pass = document.getElementById('auth-pass').value;
    const errEl = document.getElementById('auth-error');
    const btn = document.getElementById('auth-submit-btn');
    if (!email || !pass) { errEl.textContent = '⚠ Vui lòng nhập email và mật khẩu.'; return; }
    if (mode === 'register') {
        const name = document.getElementById('auth-name').value.trim();
        if (!name) { errEl.textContent = '⚠ Vui lòng nhập họ tên.'; return; }
        if (pass.length < 6) { errEl.textContent = '⚠ Mật khẩu tối thiểu 6 ký tự.'; return; }
        btn.textContent = 'Đang tạo...'; btn.disabled = true;
        try {
            const res = await fetch('/register', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name, email, password:pass}) });
            const data = await res.json();
            btn.textContent = 'Tạo tài khoản →'; btn.disabled = false;
            if (data.success) { closeAuthModal(); location.reload(); } else { errEl.textContent = '⚠ ' + data.error; }
        } catch(e) { btn.textContent = 'Tạo tài khoản →'; btn.disabled = false; errEl.textContent = '⚠ Lỗi kết nối.'; }
    } else {
        btn.textContent = 'Đang đăng nhập...'; btn.disabled = true;
        try {
            const res = await fetch('/login', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, password:pass}) });
            const data = await res.json();
            btn.textContent = 'Đăng nhập →'; btn.disabled = false;
            if (data.success) { closeAuthModal(); location.reload(); } else { errEl.textContent = '⚠ ' + data.error; }
        } catch(e) { btn.textContent = 'Đăng nhập →'; btn.disabled = false; errEl.textContent = '⚠ Lỗi kết nối.'; }
    }
}

function openSupportModal(type) {
    const existing = document.getElementById('support-modal');
    if (existing) existing.remove();
    const titles = { 'faq': 'Câu hỏi thường gặp', 'privacy': 'Chính sách bảo mật', 'terms': 'Điều khoản sử dụng' };
    const content = {
        'faq': '<h3>Làm sao để dự đoán giá?</h3><p>Nhập thông tin bất động sản vào form, hệ thống AI sẽ phân tích và trả về mức giá ước tính.</p><h3>Độ chính xác?</h3><p>Mô hình đạt trên 85% trên tập kiểm tra.</p>',
        'privacy': '<p>ProphetEstate cam kết bảo vệ quyền riêng tư. Chúng tôi không bán hoặc chia sẻ thông tin cá nhân.</p>',
        'terms': '<p>Kết quả dự đoán chỉ mang tính tham khảo, không thay thế tư vấn chuyên gia bất động sản.</p>'
    };
    const modal = document.createElement('div');
    modal.id = 'support-modal';
    modal.className = 'auth-modal-overlay';
    modal.innerHTML = `
        <div class="auth-modal-box" onclick="event.stopPropagation()">
            <button class="auth-modal-close" onclick="closeSupportModal()">✕</button>
            <h2 class="auth-modal-title">${titles[type] || 'Thông tin'}</h2>
            <div style="margin-top:1.5rem;color:var(--colors-body);line-height:1.7">${content[type] || 'Đang cập nhật.'}</div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => { if (e.target === modal) closeSupportModal(); });
    requestAnimationFrame(() => { modal.classList.add('active'); });
}

function closeSupportModal() {
    const modal = document.getElementById('support-modal');
    if (!modal) return;
    modal.classList.remove('active');
    setTimeout(() => modal.remove(), 300);
}

/* ==========================================
   GALLERY FUNCTIONS
   ========================================== */

function galleryNav(direction) {
    const imgs = document.querySelectorAll('#gallery-main img');
    const thumbs = document.querySelectorAll('#gallery-thumbs img');
    let currentIdx = Array.from(imgs).findIndex(img => img.classList.contains('active'));
    let newIdx = (currentIdx + direction + imgs.length) % imgs.length;
    galleryGoto(newIdx);
}

function galleryGoto(index) {
    const imgs = document.querySelectorAll('#gallery-main img');
    const thumbs = document.querySelectorAll('#gallery-thumbs img');
    const counter = document.getElementById('gallery-counter');
    
    imgs.forEach(img => img.classList.remove('active'));
    thumbs.forEach(thumb => thumb.classList.remove('active'));
    
    if (imgs[index]) imgs[index].classList.add('active');
    if (thumbs[index]) thumbs[index].classList.add('active');
    if (counter) counter.textContent = `${index + 1} / ${imgs.length}`;
}

/* ==========================================
   PROPERTY DETAIL FUNCTIONS
   ========================================== */

function revealPhone() {
    const phoneDisplay = document.getElementById('phone-display');
    const phoneRevealBtn = document.querySelector('.phone-reveal-btn');
    const fullPhone = phoneDisplay.getAttribute('data-full-phone');
    
    if (phoneDisplay && fullPhone) {
        phoneDisplay.textContent = fullPhone;
        if (phoneRevealBtn) phoneRevealBtn.remove();
    }
}

function handleContact(event) {
    event.preventDefault();
    const phoneDisplay = document.getElementById('phone-display');
    const fullPhone = phoneDisplay.getAttribute('data-full-phone');
    
    if (fullPhone) {
        alert(`Số điện thoại liên hệ: ${fullPhone}\n\nBạn sẽ được chuyển hướng để liên hệ...`);
    } else {
        alert('Vui lòng đăng nhập để xem số điện thoại');
    }
}
