import numpy as np


def get_user_input():
    """دریافت ورودی از کاربر"""
    print("=" * 60)
    print("حل دستگاه معادلات خطی با روش گاوس-سیدل")
    print("=" * 60)

    # دریافت تعداد معادلات
    n = int(input("تعداد معادلات (n) را وارد کنید: "))

    # دریافت ماتریس A
    print(f"\nماتریس ضرایب A ({n}×{n}) را وارد کنید:")
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_input = input(f"سطر {i + 1} (با فاصله جدا کنید): ")
        A[i] = list(map(float, row_input.split()))

    # دریافت بردار b
    print(f"\nبردار سمت راست b ({n}×1) را وارد کنید:")
    b_input = input("عناصر بردار (با فاصله جدا کنید): ")
    b = np.array(list(map(float, b_input.split())))

    # دریافت حدس اولیه
    use_default_x0 = input("\nآیا می‌خواهید از حدس اولیه صفر استفاده کنید؟ (y/n): ").lower()
    if use_default_x0 == 'y':
        x0 = np.zeros(n)
    else:
        x0_input = input(f"حدس اولیه x0 ({n}×1) را وارد کنید (با فاصله جدا کنید): ")
        x0 = np.array(list(map(float, x0_input.split())))

    # دریافت دقت بر اساس رقم اعشار
    decimal_places = int(input("دقت بر اساس چند رقم اعشار؟ (مثلاً 6): "))
    tol = 10 ** (-decimal_places)

    # دریافت پارامتر omega برای SOR
    omega = float(input("پارامتر آرامش (omega) را وارد کنید (برای گاوس-سیدل: 1): "))

    # حداکثر تعداد تکرار
    max_iter = int(input("حداکثر تعداد تکرار: "))

    print("=" * 60)
    return A, b, x0, tol, max_iter, omega, decimal_places


def calculate_norms(B):
    """محاسبه نرم‌های ماتریس B"""
    norm_1 = np.linalg.norm(B, ord=1)  # نرم 1
    norm_2 = np.linalg.norm(B, ord=2)  # نرم 2
    norm_inf = np.linalg.norm(B, ord=np.inf)  # نرم بی‌نهایت
    return norm_1, norm_2, norm_inf


def check_convergence(B, A):
    """بررسی همگرایی با استفاده از نرم‌ها و شعاع طیفی"""
    # محاسبه نرم‌ها
    norm_1, norm_2, norm_inf = calculate_norms(B)

    # محاسبه شعاع طیفی
    eigenvalues = np.linalg.eigvals(B)
    spectral_radius = np.max(np.abs(eigenvalues))

    # بررسی غالب قطری
    n = len(A)
    is_diag_dom = True
    for i in range(n):
        diag_val = abs(A[i, i])
        sum_row = np.sum(np.abs(A[i, :])) - diag_val
        if diag_val <= sum_row:
            is_diag_dom = False
            break

    # شرایط همگرایی
    convergence_conditions = {
        'norm_1_condition': norm_1 < 1,
        'norm_2_condition': norm_2 < 1,
        'norm_inf_condition': norm_inf < 1,
        'spectral_radius_condition': spectral_radius < 1,
        'diagonal_dominance': is_diag_dom
    }

    # آیا همگرا می‌شود؟
    will_converge = (norm_1 < 1 or norm_2 < 1 or norm_inf < 1 or
                     spectral_radius < 1 or is_diag_dom)

    return norm_1, norm_2, norm_inf, spectral_radius, is_diag_dom, will_converge, convergence_conditions


def gauss_seidel_matrix_form(A, b, x0, tol, max_iter, omega, decimal_places):
    """
    حل دستگاه با روش گاوس-سیدل به فرم ماتریسی
    """
    n = len(A)

    # 1. ماتریس‌های D, L, U
    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)  # پایین‌مثلثی (بدون قطر)
    U = np.triu(A, k=1)  # بالا‌مثلثی (بدون قطر)

    print("\n" + "=" * 60)
    print("مرحله 1: شکستن ماتریس A به D, L, U")
    print("=" * 60)
    print(f"ماتریس A:\n{np.round(A, decimal_places)}")
    print(f"\nماتریس D (قطری):\n{np.round(D, decimal_places)}")
    print(f"\nماتریس L (پایین‌مثلثی):\n{np.round(L, decimal_places)}")
    print(f"\nماتریس U (بالا‌مثلثی):\n{np.round(U, decimal_places)}")

    # 2. محاسبه ماتریس تکرار B و بردار c
    D_omega_L = D + omega * L
    inv_D_omega_L = np.linalg.inv(D_omega_L)
    B = inv_D_omega_L @ ((1 - omega) * D - omega * U)
    c = omega * inv_D_omega_L @ b

    print("\n" + "=" * 60)
    print("مرحله 2: محاسبه ماتریس تکرار B و بردار c")
    print("=" * 60)
    print(f"ماتریس (D + ωL):\n{np.round(D_omega_L, decimal_places)}")
    print(f"\nمعکوس (D + ωL):\n{np.round(inv_D_omega_L, decimal_places)}")
    print(f"\nماتریس تکرار B:\n{np.round(B, decimal_places)}")
    print(f"\nبردار c:\n{np.round(c, decimal_places)}")

    # 3. بررسی همگرایی قبل از شروع محاسبات
    print("\n" + "=" * 60)
    print("مرحله 3: بررسی همگرایی")
    print("=" * 60)

    norm_1, norm_2, norm_inf, spectral_radius, is_diag_dom, will_converge, conditions = check_convergence(B, A)

    print(f"نرم 1 ماتریس B: {norm_1:.{decimal_places}f}")
    print(f"نرم 2 ماتریس B: {norm_2:.{decimal_places}f}")
    print(f"نرم بی‌نهایت ماتریس B: {norm_inf:.{decimal_places}f}")
    print(f"شعاع طیفی ماتریس B (ρ(B)): {spectral_radius:.{decimal_places}f}")
    print(f"آیا ماتریس A غالب قطری است؟ {is_diag_dom}")

    print("\nشرایط همگرایی:")
    print(f"  • نرم 1 < 1: {norm_1:.{decimal_places}f} < 1 → {'✅ برقرار' if conditions['norm_1_condition'] else '❌ برقرار نیست'}")
    print(f"  • نرم 2 < 1: {norm_2:.{decimal_places}f} < 1 → {'✅ برقرار' if conditions['norm_2_condition'] else '❌ برقرار نیست'}")
    print(f"  • نرم بی‌نهایت < 1: {norm_inf:.{decimal_places}f} < 1 → {'✅ برقرار' if conditions['norm_inf_condition'] else '❌ برقرار نیست'}")
    print(f"  • شعاع طیفی < 1: {spectral_radius:.{decimal_places}f} < 1 → {'✅ برقرار' if conditions['spectral_radius_condition'] else '❌ برقرار نیست'}")
    print(f"  • ماتریس غالب قطری: {'✅ هست' if conditions['diagonal_dominance'] else '❌ نیست'}")

    if not will_converge:
        print("\n❌ هشدار: با توجه به معیارهای همگرایی، این دستگاه ممکن است همگرا نباشد!")
        continue_anyway = input("آیا می‌خواهید ادامه دهید؟ (y/n): ").lower()
        if continue_anyway != 'y':
            print("محاسبات متوقف شد.")
            return None, None, None, None, None, None, None
    else:
        print("\n✅ شرایط همگرایی برقرار است.")

    # 4. انجام تکرارها
    print("\n" + "=" * 60)
    print("مرحله 4: شروع تکرارها")
    print("=" * 60)
    print(f"حدس اولیه: {np.round(x0, decimal_places)}")
    print(f"دقت: {tol} ({decimal_places} رقم اعشار)")
    print(f"حداکثر تکرار: {max_iter}")
    print("-" * 60)

    x = x0.copy()
    history = [x0.copy()]
    iteration_details = []  # برای ذخیره جزئیات هر تکرار

    print(f"{'Iter':<5} {'x':<40} {'Bx + c':<40} {'Error':<15}")
    print("-" * 100)

    for k in range(1, max_iter + 1):
        # محاسبه جدید: x_new = B * x + c
        x_new = B @ x + c

        # محاسبه خطا
        error = np.linalg.norm(x_new - x, ord=np.inf)

        # ذخیره تاریخچه
        history.append(x_new.copy())

        # محاسبه Bx + c برای نمایش
        Bx_plus_c = B @ x + c

        # ذخیره جزئیات این تکرار
        iteration_details.append({
            'iteration': k,
            'x_old': x.copy(),
            'x_new': x_new.copy(),
            'Bx_plus_c': Bx_plus_c.copy(),
            'error': error
        })

        # نمایش اطلاعات
        if k <= 10 or k % 10 == 0:  # 10 تکرار اول و سپس هر 10 تکرار
            print(f"{k:<5} {str(np.round(x_new, decimal_places)):<40} "
                  f"{str(np.round(Bx_plus_c, decimal_places)):<40} "
                  f"{error:.{decimal_places}e}")

        # بررسی همگرایی
        if error < tol:
            print("\n" + "=" * 60)
            print(f"✅ همگرایی در تکرار {k} با خطای {error:.{decimal_places}e} < {tol}")
            break

        x = x_new.copy()
    else:
        print("\n" + "=" * 60)
        print(f"⛔ حداکثر تکرار ({max_iter}) رسید. خطای نهایی: {error:.{decimal_places}e}")

    # 5. نمایش نتایج نهایی
    print("\n" + "=" * 60)
    print("نتایج نهایی")
    print("=" * 60)

    print(f"✳️  تعداد تکرارها: {len(history) - 1}")
    print(f"✳️  جواب نهایی (x): {np.round(x, decimal_places)}")

    # بررسی صحت جواب
    Ax = A @ x
    print(f"✳️  بررسی: A * x = {np.round(Ax, decimal_places)}")
    print(f"✳️  بردار b اصلی: {np.round(b, decimal_places)}")

    residual = np.linalg.norm(Ax - b)
    print(f"✳️  باقیمانده (Residual): {residual:.{decimal_places}e}")

    print("\n✳️  جمع‌بندی نرم‌ها:")
    print(f"   نرم 1 ماتریس B: {norm_1:.{decimal_places}f}")
    print(f"   نرم 2 ماتریس B: {norm_2:.{decimal_places}f}")
    print(f"   نرم بی‌نهایت ماتریس B: {norm_inf:.{decimal_places}f}")
    print(f"   شعاع طیفی ماتریس B: {spectral_radius:.{decimal_places}f}")
    print(f"   غالب قطری: {'بله' if is_diag_dom else 'خیر'}")

    print("=" * 60)

    # برگرداندن جزئیات تکرارهای خاص
    if len(iteration_details) >= 3:
        print("\nنمایش 3 تکرار اول:")
        for i in range(min(3, len(iteration_details))):
            detail = iteration_details[i]
            print(f"\nتکرار {detail['iteration']}:")
            print(f"  x_old: {np.round(detail['x_old'], decimal_places)}")
            print(f"  Bx + c: {np.round(detail['Bx_plus_c'], decimal_places)}")
            print(f"  x_new: {np.round(detail['x_new'], decimal_places)}")
            print(f"  خطا: {detail['error']:.{decimal_places}e}")

    print("=" * 60)

    return x, history, B, {'norm_1': norm_1, 'norm_2': norm_2, 'norm_inf': norm_inf}, \
           spectral_radius, is_diag_dom, iteration_details


def main():
    """تابع اصلی"""
    try:
        # دریافت ورودی از کاربر
        A, b, x0, tol, max_iter, omega, decimal_places = get_user_input()

        # حل با روش گاوس-سیدل
        result = gauss_seidel_matrix_form(A, b, x0, tol, max_iter, omega, decimal_places)

        if result[0] is not None:
            x, history, B, norms, spectral_radius, is_diag_dom, iteration_details = result

            # ذخیره نتایج در فایل (اختیاری)
            save_results = input("\nآیا می‌خواهید نتایج را در فایل ذخیره کنید؟ (y/n): ").lower()
            if save_results == 'y':
                filename = input("نام فایل را وارد کنید (بدون پسوند): ") + ".txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("نتایج حل دستگاه معادلات خطی با روش گاوس-سیدل\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"ماتریس A:\n{A}\n\n")
                    f.write(f"بردار b: {b}\n\n")
                    f.write(f"حدس اولیه: {x0}\n\n")
                    f.write(f"جواب نهایی: {x}\n\n")
                    f.write(f"تعداد تکرارها: {len(history) - 1}\n")
                    f.write(f"دقت: {tol}\n")
                    f.write(f"نرم 1 ماتریس B: {norms['norm_1']}\n")
                    f.write(f"نرم 2 ماتریس B: {norms['norm_2']}\n")
                    f.write(f"نرم بی‌نهایت ماتریس B: {norms['norm_inf']}\n")
                    f.write(f"شعاع طیفی ماتریس B: {spectral_radius}\n")
                    f.write(f"غالب قطری: {is_diag_dom}\n")
                print(f"✅ نتایج در فایل '{filename}' ذخیره شد.")

        # امکان حل دستگاه دیگر
        another = input("\nآیا می‌خواهید دستگاه دیگری را حل کنید؟ (y/n): ").lower()
        if another == 'y':
            main()

    except ValueError as e:
        print(f"خطا در ورودی: {e}")
    except np.linalg.LinAlgError as e:
        print(f"خطا در محاسبات ماتریسی: {e}")
    except Exception as e:
        print(f"خطای ناشناخته: {e}")


if __name__ == "__main__":
    main()