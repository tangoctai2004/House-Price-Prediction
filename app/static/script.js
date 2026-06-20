document.addEventListener('DOMContentLoaded', () => {
    const typeRadios = document.querySelectorAll('input[name="property_type_toggle"]');
    const chungCuFields = document.querySelectorAll('.chung-cu-only');
    const nhaDatFields = document.querySelectorAll('.nha-dat-only');
    const form = document.getElementById('prediction-form');
    const predictionModal = document.getElementById('prediction-modal');
    const predictedPriceEl = document.getElementById('predicted-price');
    const resultMessageEl = document.getElementById('result-message');
    const errorEl = document.getElementById('error-message');
    const submitBtn = document.getElementById('submit-btn');
    // Fix white screen on back button (bfcache)
    window.addEventListener('pageshow', (event) => {
        document.body.classList.remove('page-transitioning');
    });

    // Helper for smooth transitions
    function handleLinkClick(e) {
        const link = e.target.closest('a');
        if (link && link.href && link.target !== '_blank' && 
            !link.href.startsWith('#') && !link.href.startsWith('javascript:') && 
            !link.getAttribute('onclick') && !link.classList.contains('auth-modal-close')) {
            
            // Don't transition for logout or external
            if (link.href.includes('/logout') || !link.href.includes(window.location.host)) return;

            e.preventDefault();
            document.body.classList.add('page-transitioning');
            setTimeout(() => {
                window.location.href = link.href;
            }, 250);
        }
    }

    document.addEventListener('click', handleLinkClick);

    if (!form || !submitBtn) return;
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    const modalCloseBtn = document.getElementById('prediction-modal-close');
    const modalOverlay = document.querySelector('.prediction-modal-overlay');
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

    // Mapping Tỉnh -> Quận/Huyện
    const locationData = {
        "Hà Nội": ["Cầu Giấy", "Nam Từ Liêm", "Bắc Từ Liêm", "Tây Hồ", "Thanh Xuân", "Hà Đông", "Đống Đa", "Hoàng Mai", "Long Biên", "Ba Đình", "Hai Bà Trưng", "Gia Lâm", "Đông Anh", "Thanh Trì", "Hoài Đức"],
        "TP. Hồ Chí Minh": ["Quận 1", "Quận 2", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8", "Quận 9", "Quận 10", "Quận 11", "Quận 12", "Bình Tân", "Bình Thạnh", "Tân Phú", "Bình Chánh", "Thủ Đức", "Tân Bình", "Phú Nhuận", "Nhà Bè", "Hóc Môn", "Gò Vấp"],
        "Đà Nẵng": ["Ngũ Hành Sơn", "Sơn Trà", "Cẩm Lệ", "Liên Chiểu", "Hải Châu"],
        "Khác": ["Khác"]
    };

    const provinceSelect = document.getElementById('province');
    const districtSelect = document.getElementById('district');

    function updateDistrictSelect() {
        if (!provinceSelect || !districtSelect) return;
        const selectedProvince = provinceSelect.value;
        const districts = locationData[selectedProvince] || [];
        
        districtSelect.innerHTML = '';
        districts.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d;
            districtSelect.appendChild(opt);
        });
    }

    if (provinceSelect) {
        provinceSelect.addEventListener('change', updateDistrictSelect);
        updateDistrictSelect(); // init
    }

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

        // C1: Validation trực quan
        const areaInput = document.getElementById('area');
        const bedroomsInput = document.getElementById('bedrooms');
        let hasError = false;

        // Reset previous validation states
        form.querySelectorAll('.input-error').forEach(el => el.classList.remove('input-error'));

        if (!areaInput.value || parseFloat(areaInput.value) <= 0) {
            areaInput.classList.add('input-error');
            hasError = true;
        }
        if (!bedroomsInput.value || parseInt(bedroomsInput.value) <= 0) {
            bedroomsInput.classList.add('input-error');
            hasError = true;
        }

        if (hasError) {
            errorEl.textContent = 'Vui lòng nhập đầy đủ Diện tích và Số phòng ngủ.';
            return;
        }
        
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
                // Show modal
                if (predictionModal) predictionModal.classList.add('active');
                
                // Animate price
                const finalPrice = result.predicted_price_vnd;
                animateValue(predictedPriceEl, 0, finalPrice, 2000);

                // 1. Render Confidence Interval
                const ciEl = document.getElementById('confidence-interval');
                if (ciEl) {
                    ciEl.innerHTML = `Khoảng ước tính: ${result.price_low} — ${result.price_high} tỷ (±${result.mae})`;
                }

                const transformerBox = document.getElementById('transformer-comparison');
                const transformerPrice = document.getElementById('transformer-price');
                const transformerDiff = document.getElementById('transformer-diff');
                if (transformerBox && transformerPrice && transformerDiff && result.transformer_prediction) {
                    const t = result.transformer_prediction;
                    transformerBox.style.display = 'grid';
                    transformerPrice.textContent = `${Number(t.price_billion).toFixed(2)} tỷ VNĐ`;
                    const diff = Number(t.difference_billion);
                    const diffLabel = diff >= 0 ? `cao hơn XGBoost ${diff.toFixed(2)} tỷ` : `thấp hơn XGBoost ${Math.abs(diff).toFixed(2)} tỷ`;
                    transformerDiff.textContent = diffLabel;
                } else if (transformerBox) {
                    transformerBox.style.display = 'none';
                }

                // 2. Render Explanation (XAI)
                const explainContainer = document.getElementById('explanation-container');
                const explainList = document.getElementById('explanation-list');
                if (explainContainer && explainList && result.contributions) {
                    explainContainer.style.display = 'block';
                    explainList.innerHTML = '';
                    
                    const maxImpact = Math.max(...result.contributions.map(c => c.impact));
                    
                    result.contributions.forEach(c => {
                        const pct = (c.impact / maxImpact) * 100;
                        const item = document.createElement('div');
                        item.innerHTML = `
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
                                <span style="color: var(--colors-ink); font-weight: 500;">${c.feature}</span>
                                <span style="color: var(--colors-primary); font-weight: 600;">+${c.impact} ${c.unit}</span>
                            </div>
                            <div style="height: 6px; background: var(--colors-surface-soft); border-radius: 3px; overflow: hidden;">
                                <div style="width: 0%; height: 100%; background: var(--colors-primary); transition: width 1s ease-out; border-radius: 3px;" class="impact-bar" data-pct="${pct}"></div>
                            </div>
                        `;
                        explainList.appendChild(item);
                    });
                    
                    setTimeout(() => {
                        document.querySelectorAll('.impact-bar').forEach(bar => {
                            bar.style.width = bar.getAttribute('data-pct') + '%';
                        });
                    }, 150);
                }
                
                resultMessageEl.textContent = result.message || 'Mô hình AI đã phân tích thành công đặc điểm của bất động sản này.';

                // Render Price per m2 and District Comparison
                const pricePerM2El = document.getElementById('price-per-m2');
                if (pricePerM2El && result.price_per_m2) {
                    pricePerM2El.textContent = `(~ ${result.price_per_m2} Triệu/m²)`;
                }

                const badgeEl = document.getElementById('district-comparison-badge');
                if (badgeEl && result.district_avg_m2) {
                    badgeEl.style.display = 'inline-block';
                    const diff = result.price_per_m2 - result.district_avg_m2;
                    const pct = Math.abs(diff / result.district_avg_m2) * 100;
                    if (pct < 5) {
                        badgeEl.textContent = 'Tương đương mặt bằng chung';
                        badgeEl.style.background = 'var(--colors-surface-alt)';
                        badgeEl.style.color = 'var(--colors-ink)';
                    } else if (diff > 0) {
                        badgeEl.textContent = `Cao hơn khu vực ${pct.toFixed(0)}%`;
                        badgeEl.style.background = '#fee2e2'; // Light red
                        badgeEl.style.color = '#b91c1c'; // Dark red
                    } else {
                        badgeEl.textContent = `Thấp hơn khu vực ${pct.toFixed(0)}%`;
                        badgeEl.style.background = '#dcfce7'; // Light green
                        badgeEl.style.color = '#15803d'; // Dark green
                    }
                } else if (badgeEl) {
                    badgeEl.style.display = 'none';
                }

                // Render similar properties
                const similarContainer = document.getElementById('similar-properties-container');
                const similarList = document.getElementById('similar-list');
                similarList.innerHTML = ''; // Clear old

                if (result.similar_properties && result.similar_properties.length > 0) {
                    similarContainer.style.display = 'flex';
                    result.similar_properties.forEach(prop => {
                        const card = document.createElement('a');
                        const type = prop.property_type === 'nha_dat' ? 'nha_dat' : 'chung_cu';
                        const id = prop.id;
                        const image = prop.image;
                        const title = escapeHtml(prop.title);
                        const district = escapeHtml(prop.district);
                        const price = prop.price_billion;
                        const area = prop.area_m2;

                        card.href = `/property/${type}/${id}`;
                        card.target = '_blank';
                        card.style.cssText = 'display: flex; gap: 12px; padding: 12px; border: 1px solid var(--colors-hairline); border-radius: 8px; text-decoration: none; color: inherit; transition: border-color 0.2s; background: white; margin-bottom: 2px; flex-shrink: 0;';
                        card.onmouseover = () => card.style.borderColor = 'var(--colors-primary)';
                        card.onmouseout = () => card.style.borderColor = 'var(--colors-hairline)';

                        const imgHtml = image 
                            ? `<img src="${image}" alt="House" style="width: 80px; height: 80px; object-fit: cover; border-radius: 6px; flex-shrink: 0; background: #f0f0f0;">` 
                            : `<div style="width: 80px; height: 80px; background: #f0f0f0; border-radius: 6px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 24px;">🏠</div>`;

                        card.innerHTML = `
                            ${imgHtml}
                            <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; overflow: hidden;">
                                <h5 style="margin: 0; font-size: 0.9rem; line-height: 1.4; font-weight: 500; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; color: var(--colors-ink); text-align: left;">${title}</h5>
                                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 8px;">
                                    <span style="color: var(--colors-primary); font-weight: 600; font-size: 1.05rem;">${price} Tỷ</span>
                                    <span style="color: var(--colors-muted); font-size: 0.8rem; white-space: nowrap;">📐 ${area}m² &nbsp; 📍 ${district}</span>
                                </div>
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

    // Close modal logic
    function closeModal() {
        if (predictionModal) {
            predictionModal.classList.remove('active');
        }
        // Reset form slightly if needed, but keeping input values is better UX
        // form.reset(); 
        // updateFields();
    }

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', closeModal);
    }
    
    if (modalOverlay) {
        modalOverlay.addEventListener('click', closeModal);
    }

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
