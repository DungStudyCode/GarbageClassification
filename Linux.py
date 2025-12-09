import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_capabilities():
    fig, ax = plt.subplots(figsize=(14, 7))

    # --- PHẦN 1: MÔ HÌNH CŨ (TRADITIONAL ROOT) ---
    # Vẽ vòng tròn quyền lực (Monolithic)
    root_circle = patches.Circle((0.25, 0.6), 0.15, color='#d32f2f', alpha=0.9)
    ax.add_patch(root_circle)
    ax.text(0.25, 0.6, "FULL ROOT\n(UID 0)\nToàn quyền", ha='center', va='center', 
            color='white', fontweight='bold', fontsize=12)

    # Ứng dụng Web (Bị rủi ro)
    app_box_old = dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c62828")
    ax.text(0.25, 0.25, "Web Service\n(Cần mở port 80)", ha='center', bbox=app_box_old, fontsize=10)

    # Mũi tên trao quyền (Nguy hiểm)
    ax.annotate("Cấp toàn bộ quyền", xy=(0.25, 0.32), xytext=(0.25, 0.45),
                arrowprops=dict(arrowstyle="->", lw=3, color="#d32f2f"), ha='center')
    
    ax.text(0.25, 0.1, "RỦI RO CAO:\nNếu Web Service bị hack,\nkẻ tấn công có Full Root.", 
            ha='center', color='#b71c1c', fontsize=9, style='italic')

    # --- PHẦN 2: MÔ HÌNH CAPABILITIES (PHÂN MẢNH) ---
    # Vẽ các mảnh ghép Capabilities (Grid)
    start_x, start_y = 0.55, 0.75
    gap = 0.02
    width, height = 0.12, 0.08
    
    caps = [
        ("CAP_CHOWN", "#90a4ae"), ("CAP_KILL", "#90a4ae"), ("CAP_SYS_TIME", "#90a4ae"),
        ("CAP_NET_ADMIN", "#90a4ae"), ("CAP_SYS_BOOT", "#90a4ae"), ("CAP_AUDIT_WRITE", "#90a4ae"),
        ("CAP_NET_BIND_SERVICE", "#2e7d32"), # Highlight cái này (Màu xanh)
        ("CAP_SYS_MODULE", "#90a4ae"), ("CAP_DAC_OVERRIDE", "#90a4ae")
    ]

    # Vẽ lưới 3x3
    coords = {} # Lưu tọa độ để vẽ mũi tên
    for i, (name, color) in enumerate(caps):
        row = i // 3
        col = i % 3
        x = start_x + col * (width + gap)
        y = start_y - row * (height + gap)
        
        # Vẽ hộp Cap
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", fc=color, ec="white")
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # Lưu tọa độ của CAP_NET_BIND_SERVICE
        if name == "CAP_NET_BIND_SERVICE":
            coords['target'] = (x + width/2, y)

    ax.text(start_x + 1.5*width, start_y + 0.15, "LINUX CAPABILITIES\n(Quyền Root được chia nhỏ)", 
            ha='center', fontsize=12, fontweight='bold', color='#1565c0')

    # Ứng dụng Web (An toàn)
    app_box_new = dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32")
    ax.text(start_x + 1.5*width, 0.25, "Web Service\n(Cần mở port 80)", ha='center', bbox=app_box_new, fontsize=10)

    # Mũi tên cấp quyền (An toàn)
    # Chỉ nối đúng vào CAP_NET_BIND_SERVICE
    ax.annotate("Chỉ cấp quyền này", 
                xy=coords['target'], xytext=(start_x + 1.5*width, 0.32),
                arrowprops=dict(arrowstyle="->", lw=3, color="#2e7d32"), 
                ha='center', fontsize=9, color="#2e7d32", fontweight='bold')

    ax.text(start_x + 1.5*width, 0.1, "AN TOÀN (Least Privilege):\nNếu Web Service bị hack,\nkẻ tấn công KHÔNG thể làm gì khác.", 
            ha='center', color='#1b5e20', fontsize=9, style='italic')

    # --- KẺ ĐƯỜNG PHÂN CÁCH ---
    ax.plot([0.45, 0.45], [0.05, 0.95], color="gray", linestyle="--")

    # Trang trí
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title("Hình 3.6: Phân mảnh đặc quyền Root thành các Capabilities", fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    plt.show()

draw_capabilities()