import matplotlib
from matplotlib import patches, rcParams, transforms, colormaps
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from themes import THEMES, SCIENTIFIC_Y_LABELS, BUSINESS_Y_LABELS, SCIENTIFIC_X_LABELS, BUSINESS_X_LABELS, COMPARATIVE_LABELS, HISTOGRAM_Y_LABELS, FONT_FAMILIES, HEATMAP_XLABELS_SCIENTIFIC, HEATMAP_YLABELS_SCIENTIFIC, HEATMAP_XLABELS_BUSINESS, HEATMAP_YLABELS_BUSINESS, COLORBAR_TITLES_SCIENTIFIC, COLORBAR_TITLES_BUSINESS, HEATMAP_CHART_TITLES, HEATMAP_ANNOTATION_FORMATS

# ===================================================================================
# == DATA GENERATION & THEMES ==
# ===================================================================================

def generate_realistic_data(num_points, max_scale, allow_negative=False, pattern_type=None, domain='scientific'):
    """
    Generate statistically realistic data based on real-world scientific and business patterns.
    
    Critical improvements:
    - Domain-specific parameter constraints based on published literature
    - Heteroscedastic noise models matching measurement error characteristics
    - Realistic coefficient of variation (CV) ranges for biological systems
    - Enforced monotonicity and physical plausibility constraints
    - Measurement precision limitations
    """
    
    if pattern_type is None:
        # Weight patterns by actual frequency in scientific literature
        if domain == 'scientific':
            pattern_type = np.random.choice(
                ['dose_response', 'replicates', 'exponential_decay', 'power_law', 
                 'sigmoid_growth', 'linear_regression', 'gaussian_peak', 'enzyme_kinetics'],
                p=[0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05]
            )
        else:  # business
            pattern_type = np.random.choice(
                ['seasonal_trend', 'pareto_distribution', 'exponential_growth', 
                 'market_saturation', 'random_walk_drift', 'step_intervention'],
                p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
            )
    
    data = np.zeros(num_points)
    
    # === SCIENTIFIC PATTERNS WITH REALISTIC CONSTRAINTS ===
    if pattern_type == 'dose_response':
        # Hill equation with literature-validated parameter ranges
        log_conc = np.linspace(-10, -3, num_points)  # pM to mM range (realistic drug concentrations)
        
        # Realistic EC50 values from drug databases (ChEMBL, PubChem)
        ec50 = np.random.uniform(-8.5, -4.5)  # 3nM to 30µM range
        
        # Hill slopes from literature (rarely exceed 4, typically 0.7-2.5)
        hill_slope = np.random.choice([
            np.random.uniform(0.7, 1.3),   # 60% - physiological range
            np.random.uniform(1.3, 2.0),   # 30% - cooperative binding
            np.random.uniform(2.0, 3.5)    # 10% - strong cooperativity
        ], p=[0.6, 0.3, 0.1])
        
        # Realistic baseline and maximum response
        baseline = np.random.uniform(0, 0.08) * max_scale  # 0-8% baseline activity
        max_response = np.random.uniform(0.85, 0.98) * max_scale  # 85-98% max response
        
        # Hill equation
        data = baseline + (max_response - baseline) / (1 + 10**((ec50 - log_conc) * hill_slope))
        
        # Heteroscedastic noise: CV increases at curve inflection points
        response_fraction = (data - baseline) / (max_response - baseline)
        cv = 0.05 + 0.10 * np.sqrt(response_fraction * (1 - response_fraction))
        noise = np.random.normal(0, data * cv, num_points)
        data += noise
        
        # Enforce non-negativity for biological measurements
        data = np.clip(data, 0, max_response * 1.05)
    
    elif pattern_type == 'replicates':
        # Biological replicates with realistic technical variation
        mean_val = np.random.uniform(0.25, 0.75) * max_scale
        
        # CV based on measurement type (qPCR: 5-15%, Western: 10-25%, Cell assays: 15-35%)
        measurement_type = np.random.choice(['qpcr', 'western', 'cell_assay'], p=[0.3, 0.3, 0.4])
        cv_ranges = {
            'qpcr': (0.05, 0.15),
            'western': (0.10, 0.25), 
            'cell_assay': (0.15, 0.35)
        }
        cv = np.random.uniform(*cv_ranges[measurement_type])
        
        # Log-normal distribution for biological variability (more realistic than normal)
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean_val) - 0.5 * sigma**2
        data = np.random.lognormal(mu, sigma, num_points)
        
        # Clip to realistic bounds
        data = np.clip(data, 0, max_scale * 1.2)
    
    elif pattern_type == 'exponential_decay':
        # Realistic pharmacokinetic/radioactive decay parameters
        t = np.linspace(0, np.random.uniform(5, 20), num_points)
        
        # Half-life ranges based on actual pharmacokinetic data
        half_life = np.random.choice([
            np.random.uniform(0.5, 2),    # Fast clearance (minutes to hours)
            np.random.uniform(2, 12),     # Moderate clearance (hours)
            np.random.uniform(12, 72)     # Slow clearance (days)
        ], p=[0.3, 0.5, 0.2])
        
        decay_constant = np.log(2) / half_life
        
        initial_value = np.random.uniform(0.8, 0.95) * max_scale
        baseline = np.random.uniform(0, 0.1) * max_scale
        
        data = (initial_value - baseline) * np.exp(-decay_constant * t) + baseline
        
        # Proportional error model (higher error at higher concentrations)
        error_proportional = 0.08  # 8% proportional error
        error_additive = 0.02 * max_scale  # 2% additive error
        noise = np.random.normal(0, data * error_proportional + error_additive, num_points)
        data += noise
        
        data = np.clip(data, baseline * 0.8, initial_value * 1.1)
    
    elif pattern_type == 'enzyme_kinetics':
        # Michaelis-Menten kinetics with realistic parameters
        substrate = np.logspace(-1, 2, num_points)  # 0.1 to 100 units
        
        # Realistic Km values (µM to mM range for most enzymes)
        km = np.random.uniform(0.5, 50)
        vmax = np.random.uniform(0.7, 0.95) * max_scale
        
        # Michaelis-Menten equation
        data = (vmax * substrate) / (km + substrate)
        
        # Add realistic experimental noise (CV = 5-15% for enzyme assays)
        cv = np.random.uniform(0.05, 0.15)
        noise = np.random.normal(0, data * cv, num_points)
        data += noise
        
        data = np.clip(data, 0, vmax * 1.05)
    
    elif pattern_type == 'gaussian_peak':
        # Spectroscopy peak or chromatography data
        x = np.arange(num_points)
        
        # Peak position and width
        mu = np.random.uniform(num_points * 0.3, num_points * 0.7)
        sigma = np.random.uniform(num_points * 0.05, num_points * 0.20)
        
        amplitude = max_scale * np.random.uniform(0.80, 0.95)
        baseline = max_scale * np.random.uniform(0, 0.10)
        
        data = amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2)) + baseline
        
        # Poisson-like noise (typical for photon counting/mass spectrometry)
        noise_factor = np.sqrt(np.abs(data - baseline))
        noise = np.random.normal(0, noise_factor * 0.3, num_points)
        data += noise
        
        data = np.clip(data, baseline * 0.9, amplitude * 1.1)
    
    # === BUSINESS PATTERNS WITH REALISTIC CONSTRAINTS ===
    elif pattern_type == 'seasonal_trend':
        # Realistic business seasonality with multiple components
        x = np.arange(num_points)
        
        # Multiple seasonality (annual + quarterly + monthly if enough points)
        components = []
        
        if num_points >= 12:  # Annual seasonality
            annual_freq = 2 * np.pi / 12
            annual_amp = max_scale * np.random.uniform(0.15, 0.30)
            annual_phase = np.random.uniform(0, 2*np.pi)
            components.append(annual_amp * np.cos(annual_freq * x + annual_phase))
        
        if num_points >= 4:   # Quarterly seasonality
            quarterly_freq = 2 * np.pi / 4
            quarterly_amp = max_scale * np.random.uniform(0.08, 0.15)
            quarterly_phase = np.random.uniform(0, 2*np.pi)
            components.append(quarterly_amp * np.cos(quarterly_freq * x + quarterly_phase))
        
        # Base level with trend
        base_level = max_scale * np.random.uniform(0.3, 0.5)
        trend_slope = max_scale * np.random.uniform(-0.05, 0.20) / num_points
        trend = x * trend_slope
        
        # Combine components
        seasonal = np.sum(components, axis=0) if components else np.zeros(num_points)
        data = base_level + trend + seasonal
        
        # Business-appropriate noise (higher during peak seasons)
        noise_base = max_scale * 0.03
        noise_seasonal = np.abs(seasonal) * 0.2
        noise = np.random.normal(0, noise_base + noise_seasonal, num_points)
        data += noise
        
        data = np.clip(data, 0, max_scale * 1.5)
    
    elif pattern_type == 'pareto_distribution':
        # Pareto principle (80/20 rule) - realistic for business data
        shape = np.random.uniform(1.05, 2.5)  # Literature range for business data
        
        # Generate Pareto samples
        samples = np.random.pareto(shape, num_points) + 1
        
        # Sort in descending order for typical business visualization
        data = np.sort(samples)[::-1]
        
        # Scale to max_scale with realistic ceiling
        data = (data / data.max()) * max_scale * np.random.uniform(0.6, 0.9)
        
        # Add small multiplicative noise (log-normal)
        noise = np.random.lognormal(0, 0.10, num_points)
        data *= noise
    
    elif pattern_type == 'exponential_growth':
        # Realistic business growth with saturation
        t = np.linspace(0, 1, num_points)
        
        # Growth rates based on actual business metrics
        growth_rate = np.random.choice([
            np.random.uniform(2, 5),      # Moderate growth
            np.random.uniform(5, 10),     # High growth
            np.random.uniform(10, 15)     # Exponential phase
        ], p=[0.5, 0.3, 0.2])
        
        initial_value = max_scale * np.random.uniform(0.05, 0.20)
        
        # Exponential with eventual saturation (logistic-like)
        data = initial_value * np.exp(growth_rate * t)
        
        # Apply market saturation
        carrying_capacity = max_scale * np.random.uniform(0.8, 1.0)
        saturation_factor = 1 / (1 + (data / carrying_capacity))
        data *= saturation_factor
        
        # Business noise (proportional to current value)
        cv = np.random.uniform(0.08, 0.20)
        noise = np.random.normal(0, data * cv, num_points)
        data += noise
        
        data = np.clip(data, initial_value * 0.8, carrying_capacity * 1.1)
    
    # === FALLBACK PATTERNS ===
    elif pattern_type == 'linear':
        start = max_scale * np.random.uniform(0.10, 0.40)
        end = max_scale * np.random.uniform(0.50, 0.90)
        
        if domain == 'scientific' and np.random.random() < 0.7:
            start, end = min(start, end), max(start, end)
        
        data = np.linspace(start, end, num_points)
        
        noise_cv = np.random.uniform(0.05, 0.15)
        noise = np.random.normal(0, data * noise_cv + max_scale * 0.01, num_points)
        data += noise
    
    elif pattern_type == 'plateau':
        p1, p2 = sorted(random.sample(range(num_points + 1), 2))
        low = max_scale * np.random.uniform(0.1, 0.3)
        high = max_scale * np.random.uniform(0.7, 0.9)
        
        data = np.concatenate([
            np.full(p1, low), 
            np.full(p2 - p1, high), 
            np.full(num_points - p2, low)
        ])
        
        noise = np.random.normal(0, max_scale * 0.05, num_points)
        data += noise
    
    else:  # Default: random_walk
        start = max_scale * np.random.uniform(0.3, 0.7)
        steps = np.random.normal(0, max_scale * 0.1, num_points)
        data = start + np.cumsum(steps)
        
        noise = np.random.normal(0, max_scale * 0.05, num_points)
        data += noise
    
    # === POST-PROCESSING FOR MEASUREMENT REALISM ===
    
    # Apply measurement constraints
    if not allow_negative:
        data = np.clip(data, 0, max_scale * 1.05)  # Strict 5% overshoot for realism without excess
    else:
        data = np.clip(data, -max_scale * 0.4, max_scale * 1.05)  # Strict 5% overshoot for realism without excess
    
    # Realistic measurement precision (instruments have limited precision)
    if max_scale >= 1000:
        precision = np.random.choice([0, 1], p=[0.7, 0.3])
    elif max_scale >= 100:
        precision = np.random.choice([1, 2], p=[0.6, 0.4])
    elif max_scale >= 10:
        precision = np.random.choice([2, 3], p=[0.7, 0.3])
    else:
        precision = 3
    
    data = np.round(data, precision)
    
    # Remove impossible values
    if domain == 'scientific' and not allow_negative:
        data = np.abs(data)
    
    return data

def apply_chart_theme(ax, theme_name, orientation='vertical'):
    theme = THEMES.get(theme_name, THEMES.get('default', {}))
    if not theme:
        theme = {'facecolor': 'white', 'grid_color': '#CCCCCC', 'grid_style': 'solid', 'font': 'Arial'}
    
    ax.set_facecolor(theme.get('facecolor', 'white'))
    
    grid_axis = 'y' if orientation == 'vertical' else 'x'
    
    if theme.get('grid_style', 'none') != 'none':
        ax.grid(axis=grid_axis, color=theme.get('grid_color', '#CCCCCC'), 
                linestyle=theme.get('grid_style', 'solid'),
                linewidth=theme.get('grid_linewidth', 1.0), zorder=0)
    
    try: 
        rcParams['font.sans-serif'] = [theme.get('font', 'Arial'), 'DejaVu Sans', 'Arial']
    except Exception: 
        pass
    
    for spine, visible in theme.get('spines', {}).items():
        if spine in ax.spines: 
            ax.spines[spine].set_visible(visible)
    
    if theme.get('spine_width'):
        for spine in ax.spines.values():
            spine.set_linewidth(theme['spine_width'])
    
    if theme.get('tick_direction'): 
        ax.tick_params(axis='both', direction=theme['tick_direction'])
    
    return theme

def apply_typography_variation(ax, domain='scientific'):
    """
    Aplica diversas configurações de tipografia com base na análise do usuário.
    Varia a família da fonte, tamanhos, peso e rotação.
    """
    
    # Seleciona a família da fonte
    if domain == 'scientific':
        family = np.random.choice(['sans-serif', 'serif'], p=[0.7, 0.3])
    else: # business
        family = np.random.choice(['sans-serif', 'serif'], p=[0.8, 0.2])
    
    # Seleciona um nome de fonte específico da família escolhida
    font_name = np.random.choice(FONT_FAMILIES[family])
    
    # Tamanhos de fonte
    title_size = np.random.randint(12, 17)
    label_size = np.random.randint(10, 14)
    tick_size = np.random.randint(8, 12)
    
    try:
        # Aplica aos elementos dos eixos
        if ax.title:
            ax.title.set_fontsize(title_size)
            ax.title.set_fontfamily(font_name)
            ax.title.set_fontweight(np.random.choice(['normal', 'bold'], p=[0.6, 0.4]))
        
        if ax.xaxis.label:
            ax.xaxis.label.set_fontsize(label_size)
            ax.xaxis.label.set_fontfamily(font_name)
            
        if ax.yaxis.label:
            ax.yaxis.label.set_fontsize(label_size)
            ax.yaxis.label.set_fontfamily(font_name)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick_size)
            label.set_fontfamily(font_name)
        
        # Variações de rotação de texto
        if np.random.random() < 0.3: # 30% de chance
            rotation = np.random.choice([0, 45, 90], p=[0.5, 0.3, 0.2])
            # Gira apenas os ticks do eixo x para evitar eixo y ilegível
            ax.tick_params(axis='x', labelrotation=rotation)
            
    except Exception as e:
        # Captura erros caso as fontes não sejam encontradas no sistema
        if domain == 'scientific': # Checa contra uma variável usada no bloco try
            print(f"Aviso: Não foi possível aplicar a fonte {font_name}. Erro: {e}")
        pass # Continua com as fontes padrão

def apply_axis_scaling(ax, data_min=None, orientation='vertical', scale_type='auto'):
    """Aplica diferentes escalas de eixo (log, symlog) com base na análise."""
    
    if scale_type == 'auto':
        scale_type = np.random.choice(['linear', 'log', 'symlog'],
                                      p=[0.80, 0.15, 0.05])
    
    if scale_type == 'linear':
        return  # Não faz nada
    
    # CRÍTICO: Verifica os limites dos dados antes de aplicar a escala log
    if scale_type == 'log':
        # Se data_min não foi fornecido ou é <= 0, não podemos usar 'log'.
        # Muda para 'symlog', que lida com valores zero e negativos.
        if data_min is None or data_min <= 0:
            scale_type = 'symlog'
    
    try:
        if orientation == 'vertical':
            if scale_type == 'log':
                # Isso só será executado se data_min > 0
                ax.set_yscale('log')
            elif scale_type == 'symlog':
                # Usa linthresh 1.0 conforme sugerido na análise
                ax.set_yscale('symlog', linthresh=1.0)
        else:  # horizontal
            if scale_type == 'log':
                ax.set_xscale('log')
            elif scale_type == 'symlog':
                ax.set_xscale('symlog', linthresh=1.0)
    except Exception as e:
        # Captura quaisquer erros restantes
        print(f"AVISO: Não foi possível aplicar a escala de eixo '{scale_type}'. Erro: {e}")

# ===================================================================================
# == CHART ELEMENT ADDITIONS (CRITICAL FOR YOLO ANNOTATION) ==
# ===================================================================================

def add_bar_shadows(ax, bars, fig):
    """Add realistic drop shadows to bars"""
    for bar in bars:
        dx, dy = 2 / fig.dpi, -2 / fig.dpi
        shadow_transform = ax.transData + transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        shadow = patches.Rectangle(bar.get_xy(), bar.get_width(), bar.get_height(),
                                 transform=shadow_transform, facecolor='black',
                                 alpha=0.2, zorder=bar.get_zorder() - 0.1)
        ax.add_patch(shadow)

def add_significance_markers(ax, bar_info, y_max, orientation='vertical', error_tops=None):
    """Add statistical significance markers between bars"""
    annotations = []
    
    if len(bar_info) < 2 or random.random() < 0.4: 
        return annotations
    
    mode = random.choice(['bracket', 'letters'])
    
    if mode == 'bracket':
        try:
            idx1, idx2 = random.sample(range(len(bar_info)), 2)
        except ValueError:
            return annotations
        
        bracket_style = random.choice(['standard', 'extended'])
        
        # Get bar positions
        pos1, pos2 = bar_info[idx1]['center'], bar_info[idx2]['center']
        text = random.choice(['*', '**', '***', 'ns'])
        
        start_idx = min(idx1, idx2)
        end_idx = max(idx1, idx2)
        
        max_height_in_range = 0
        
        if orientation == 'vertical':
            # Find maximum height of ANY bar or error bar within range
            for i in range(start_idx, end_idx + 1):
                bar_height = error_tops[i] if error_tops and i < len(error_tops) else bar_info[i]['height']
                if abs(bar_height) > max_height_in_range:
                    max_height_in_range = abs(bar_height)
            
            y_for_level = max_height_in_range
            level = max_height_in_range * (1 + random.uniform(0.10, 0.20))
            
            height1 = error_tops[idx1] if error_tops and idx1 < len(error_tops) else bar_info[idx1]['height']
            height2 = error_tops[idx2] if error_tops and idx2 < len(error_tops) else bar_info[idx2]['height']
            
            if bracket_style == 'extended':
                gap = y_for_level * 0.05
                start_y1 = height1 + gap
                start_y2 = height2 + gap
                ax.plot([pos1, pos1, pos2, pos2], [start_y1, level, level, start_y2], 
                       lw=1.2, c='black', zorder=15)
            else:  # 'standard'
                tip_height = y_for_level * 0.05
                ax.plot([pos1, pos1, pos2, pos2], 
                       [level - tip_height, level, level, level - tip_height], 
                       lw=1.2, c='black', zorder=15)
            
            txt = ax.text((pos1 + pos2) / 2, level, text, ha='center', va='bottom', 
                         color='black', fontsize=12, zorder=15)
        
        else:  # Horizontal orientation
            # Find maximum "height" (width) of ANY bar in range
            for i in range(start_idx, end_idx + 1):
                bar_width = error_tops[i] if error_tops and i < len(error_tops) else bar_info[i]['height']
                if abs(bar_width) > max_height_in_range:
                    max_height_in_range = abs(bar_width)
            
            x_for_level = max_height_in_range
            level = x_for_level * (1 + random.uniform(0.15, 0.30))
            
            height1 = error_tops[idx1] if error_tops and idx1 < len(error_tops) else bar_info[idx1]['height']
            height2 = error_tops[idx2] if error_tops and idx2 < len(error_tops) else bar_info[idx2]['height']
            
            if bracket_style == 'extended':
                gap = x_for_level * 0.05
                start_x1 = height1 + gap
                start_x2 = height2 + gap
                ax.plot([start_x1, level, level, start_x2], [pos1, pos1, pos2, pos2], 
                       lw=1.2, c='black', zorder=15)
            else:  # 'standard'
                tip_height = x_for_level * 0.05
                ax.plot([level - tip_height, level, level, level - tip_height], 
                       [pos1, pos1, pos2, pos2], lw=1.2, c='black', zorder=15)
            
            txt = ax.text(level, (pos1 + pos2) / 2, text, ha='left', va='center', 
                         color='black', fontsize=12, zorder=15)
        
        annotations.append(txt)
    
    elif mode == 'letters':
        letters = random.sample(['a', 'b', 'c', 'd'], k=min(len(bar_info), 4))
        
        for i, info in enumerate(bar_info):
            if i >= len(letters): 
                break
            
            pos, height = info['center'], info['height']
            base_y = error_tops[i] if error_tops and i < len(error_tops) else height
            offset = 0.05 * y_max
            y_pos = base_y + offset if height >= 0 else base_y - offset
            va = 'bottom' if height >= 0 else 'top'
            
            if orientation == 'vertical':
                txt = ax.text(pos, y_pos, letters[i], ha='center', va=va, fontsize=10)
            else:
                txt = ax.text(y_pos, pos, letters[i], ha='left', va='center', fontsize=10)
            
            annotations.append(txt)
    
    return annotations

def apply_legend_variation(ax, num_items):
    """Aplica diversas configurações de posicionamento e estilo de legenda."""
    
    # Posições internas
    inside_locs = ['upper right', 'upper left', 'lower left', 'lower right', 
                   'center', 'center right', 'center left']
    # Posição externa (à direita)
    outside_right = 'center left'
    
    # Escolhe a localização
    if num_items <= 4:
        # Legendas pequenas podem ir para dentro
        loc = np.random.choice(inside_locs + [outside_right], 
                             p=[0.12]*7 + [0.16])
    else:
        # Legendas grandes são melhores do lado de fora
        loc = np.random.choice(inside_locs + [outside_right],
                             p=[0.05]*7 + [0.65])
    
    # Configurações de moldura
    frameon = np.random.choice([True, False], p=[0.4, 0.6])
    
    legend = None
    
    if loc == 'center left': # Trata como "fora à direita"
        # Usa bbox_to_anchor para mover a legenda para fora do eixo
        legend = ax.legend(loc=loc, bbox_to_anchor=(1.04, 0.5), frameon=frameon)
    else:
        legend = ax.legend(loc=loc, frameon=frameon)
    
    # Múltiplas colunas para muitos itens
    if num_items > 6:
        ncol = np.random.choice([1, 2], p=[0.7, 0.3])
        if legend:
            legend._ncol = ncol # Define o número de colunas
    
    return legend

def apply_pie_label_strategy(data, labels_text):
    """Implementa diversas estratégias de rotulagem para gráficos de pizza."""
    
    # Escolhe uma estratégia de rotulagem aleatória
    strategy = np.random.choice(
        ['default_leader', 'outside_pct_only', 'inside_pct_only', 'none'],
        p=[0.40, 0.30, 0.20, 0.10]
    )
    
    pie_params = {}

    if strategy == 'default_leader':
        # Estratégia 1: Rótulos de texto fora, porcentagens dentro (o seu original)
        pie_params['labels'] = labels_text
        pie_params['autopct'] = '%1.1f%%'
        pie_params['pctdistance'] = 0.7  # Porcentagem dentro
        pie_params['labeldistance'] = 1.1 # Rótulo de texto fora
    
    elif strategy == 'outside_pct_only':
        # Estratégia 2: Apenas porcentagens, fora da fatia. Sem rótulos de texto.
        pie_params['labels'] = None
        pie_params['autopct'] = '%1.1f%%'
        pie_params['pctdistance'] = 0.8 # Um pouco mais longe do centro
        pie_params['labeldistance'] = 1.15 # Posição da porcentagem
    
    elif strategy == 'inside_pct_only':
        # Estratégia 3: Apenas porcentagens, dentro da fatia. Sem rótulos de texto.
        pie_params['labels'] = None
        pie_params['autopct'] = '%1.1f%%'
        pie_params['pctdistance'] = 0.5 # Bem dentro
        pie_params['labeldistance'] = 1.1 # (Não usado)

    elif strategy == 'none':
        # Estratégia 4: Sem rótulos
        pie_params['labels'] = None
        pie_params['autopct'] = None
    
    return pie_params

def add_error_bars(ax, bar_info, orientation='vertical', measurement_type='biological'):
    """Add realistic error bars with measurement-specific characteristics"""
    error_artists = []
    error_tops = [info['height'] for info in bar_info]  # Initialize with bar heights
    
    for i, info in enumerate(bar_info):
        if random.random() < 0.7 and info['height'] >= 0:  # 70% chance for error bars
            center, value = info['center'], info['height']
            
            # Realistic error bar calculation based on measurement type
            if measurement_type == 'biological':
                # Biological replicates: SEM or SD
                error_type = np.random.choice(['sem', 'sd'], p=[0.7, 0.3])
                n_replicates = np.random.choice([3, 4, 5, 6, 8], p=[0.4, 0.3, 0.15, 0.10, 0.05])
                
                # CV based on assay type
                cv = np.random.uniform(0.10, 0.30)  # 10-30% CV for biological data
                sd = value * cv
                
                if error_type == 'sem':
                    error = sd / np.sqrt(n_replicates)
                else:
                    error = sd
                    
            elif measurement_type == 'analytical':
                # Analytical chemistry: typically smaller errors
                cv = np.random.uniform(0.02, 0.08)  # 2-8% CV
                error = value * cv
                
            elif measurement_type == 'survey':
                # Survey data: confidence intervals
                error = value * np.random.uniform(0.05, 0.15)  # 5-15% margin of error
                
            else:  # Default
                error = value * np.random.uniform(0.08, 0.20)
            
            # Create error bar
            if orientation == 'vertical':
                artist = ax.errorbar(center, value, yerr=error, fmt='none', 
                                   ecolor='black', capsize=4, elinewidth=1.2, 
                                   capthick=1.2, zorder=10)
                error_tops[i] = value + error
            else:
                artist = ax.errorbar(value, center, xerr=error, fmt='none', 
                                   ecolor='black', capsize=4, elinewidth=1.2, 
                                   capthick=1.2, zorder=10)
                error_tops[i] = value + error
                
            error_artists.append(artist)
    
    return error_artists, error_tops

def add_data_labels(ax, artists, orientation='vertical', chart_type='bar', 
                   error_tops=None, bar_info_list=None):
    """
    Add data labels to chart elements with correct positioning for stacked bars.
    
    CRITICAL FIX: Uses bar_info_list metadata to correctly position labels
    on stacked bar segments and match with error bars.
    """
    labels = []
    
    # Create lookup for bar info if available
    bar_info_centers = None
    if bar_info_list:
        bar_info_centers = [b['center'] for b in bar_info_list]
    
    for artist in artists:
        if isinstance(artist, patches.Rectangle) and chart_type in ['bar', 'histogram']:
            matched_idx = None
            matched_bar_info = None
            
            # Determine artist's center in data coordinates
            if orientation == 'vertical':
                artist_center = artist.get_x() + artist.get_width() / 2.0
                artist_height = artist.get_height()
                artist_bottom = artist.get_y()
            else:  # horizontal
                artist_center = artist.get_y() + artist.get_height() / 2.0
                artist_height = artist.get_width()
                artist_bottom = artist.get_x()
            
            # Match to bar_info using center AND bottom position (critical for stacked bars)
            if bar_info_list:
                min_distance = float('inf')
                for idx, b_info in enumerate(bar_info_list):
                    # Check center match
                    center_diff = abs(b_info['center'] - artist_center)
                    
                    # Check bottom position match (critical for stacked bars)
                    bottom_diff = abs(b_info.get('bottom', 0) - artist_bottom)
                    
                    # Combined distance metric
                    distance = center_diff + bottom_diff
                    
                    if distance < min_distance:
                        min_distance = distance
                        matched_idx = idx
                        matched_bar_info = b_info
            
            # Use segment height as label value (NOT cumulative for stacked)
            if matched_bar_info:
                label_value = matched_bar_info['height']
            else:
                if orientation == 'vertical':
                    label_value = artist_height if artist.get_y() >= 0 else -artist_height
                else:  # horizontal
                    label_value = artist_height if artist.get_x() >= 0 else -artist_height
            
            if abs(label_value) < 0.01: 
                continue
            
            label_text = f'{label_value:.1f}'
            
            # Position label at TOP of segment
            if orientation == 'vertical':
                x_pos = artist_center
                offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                
                if label_value >= 0:
                    # Use error bar top as anchor if available
                    anchor_y = error_tops[matched_idx] if matched_idx is not None and error_tops else (matched_bar_info['height'] if matched_bar_info else artist.get_y() + artist.get_height())
                    y_pos = anchor_y + offset
                    va = 'bottom'
                else:  # Negative bars
                    anchor_y = artist.get_y()
                    y_pos = anchor_y - offset
                    va = 'top'
                
                txt = ax.text(x_pos, y_pos, label_text, ha='center', va=va, 
                             fontsize=7, zorder=12)
                labels.append(txt)
            
            else:  # horizontal
                y_pos = artist_center
                offset = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
                
                if label_value >= 0:
                    anchor_x = error_tops[matched_idx] if matched_idx is not None and error_tops else (matched_bar_info['height'] if matched_bar_info else artist.get_x() + artist.get_width())
                    x_pos = anchor_x + offset
                    ha = 'left'
                else:  # Negative bars
                    anchor_x = artist.get_x()
                    x_pos = anchor_x - offset
                    ha = 'right'
                
                txt = ax.text(x_pos, y_pos, label_text, ha=ha, va='center',
                             fontsize=7, zorder=12)
                labels.append(txt)
        
        elif isinstance(artist, patches.Wedge) and chart_type == 'pie':  # For pie charts
            ang = (artist.theta2 - artist.theta1)/2. + artist.theta1
            y = np.sin(np.deg2rad(ang)); x = np.cos(np.deg2rad(ang))
            value = (artist.theta2 - artist.theta1) / 360
            
            if value > 0.05:
                txt = ax.text(0.7 * x, 0.7 * y, f'{value:.0%}', ha='center', va='center', 
                             fontsize=8, color='white', zorder=12,
                             bbox=dict(boxstyle="round,pad=0.2", fc='black', ec="none", alpha=0.4))
                labels.append(txt)
    
    return labels

def add_treatment_key_xaxis(ax, bar_info_list):
    """Add treatment key annotations below X-axis"""
    annotation_artists = []
    treatment_labels = COMPARATIVE_LABELS
    centers = [info['center'] for info in bar_info_list]
    
    if len(centers) != 4: 
        return annotation_artists  # Return empty list
    
    ax.set_xlabel(''); ax.set_xticklabels([]); ax.tick_params(axis='x', length=0)
    
    treatment1, treatment2 = random.choice(treatment_labels)
    y_pos1, y_pos2 = -0.15, -0.25
    
    text1 = ax.text(-0.1, y_pos1, f"{treatment1}", transform=ax.transAxes, 
                   ha='right', va='center', fontsize=10)
    text2 = ax.text(-0.1, y_pos2, f"{treatment2}", transform=ax.transAxes, 
                   ha='right', va='center', fontsize=10)
    
    annotation_artists.extend([text1, text2])
    
    symbols = [('-', '-'), ('+', '-'), ('-', '+'), ('+', '+')]
    blended_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    for i, center in enumerate(centers):
        ax.text(center, y_pos1, symbols[i][0], transform=blended_transform, 
               ha='center', va='center', fontsize=12)
        ax.text(center, y_pos2, symbols[i][1], transform=blended_transform, 
               ha='center', va='center', fontsize=12)
    
    return annotation_artists

# ===================================================================================
# == CHART-SPECIFIC GENERATOR FUNCTIONS (COMPLETE WITH BUG FIXES) ==
# ===================================================================================

def _generate_bar_chart(ax, theme_name, theme_config, style_config, debug_mode=False):
    """
    CRITICAL FIXES:
    - Complete stacked bar annotation metadata
    - Proper dual-axis handling
    - Consistent return structure for all code paths
    - Error bar matching for stacked segments
    """
    
    # Initialize return values at start
    data_artists = []
    other_artists = []
    bar_info_list = []
    orientation = style_config.get('orientation', 'vertical')
    error_tops = []
    axis_related_artists = []
    scale_axis_info = {'primary_scale_axis': 'y' if orientation == 'vertical' else 'x'}
    
    if debug_mode:
        print(f"DEBUG: _generate_bar_chart - Theme: {theme_name}, Style: {style_config}")
        print(f"DEBUG: _generate_bar_chart - Orientation: {orientation}")
    
    # --- DUAL Y-AXIS LOGIC ---
    if style_config.get('orientation', 'vertical') == 'vertical' and random.random() < 0.15:
        print(" - Generating dual Y-axis bar chart with scientific styles")
        is_scientific = True
        ax2 = ax.twinx()
        
        num_bars_1 = random.randint(2, 5)
        num_bars_2 = random.randint(2, 5)
        max_scale_1 = random.choice([50, 100, 200])
        max_scale_2 = random.choice([500, 1000, 2000])
        
        data_1 = generate_realistic_data(num_bars_1, max_scale_1, domain='scientific')
        data_2 = generate_realistic_data(num_bars_2, max_scale_2, domain='scientific')
        
        if debug_mode:
            print(f"DEBUG: Dual-axis chart - Data sets: {len(data_1)} and {len(data_2)} values")
            print(f"DEBUG: Dual-axis chart - Max scales: {max_scale_1} and {max_scale_2}")
        
        bar_width = 0.8
        positions_1 = np.arange(num_bars_1)
        gap = 2
        positions_2 = np.arange(num_bars_2) + num_bars_1 + gap
        all_positions = np.concatenate([positions_1, positions_2])
        
        labels_1 = [f'Cond {i+1}' for i in range(num_bars_1)]
        labels_2 = [f'Treat {i+1}' for i in range(num_bars_2)]
        all_labels = labels_1 + labels_2
        
        bar_styles = [{'facecolor': 'white', 'edgecolor': 'black', 'hatch': h} 
                     for h in ['', '////', '....', 'xxxx']]
        random.shuffle(bar_styles)
        style_1, style_2 = random.sample(bar_styles, 2)
        
        rects1 = ax.bar(positions_1, data_1, width=bar_width, zorder=2, 
                       label='Group 1', **style_1)
        rects2 = ax2.bar(positions_2, data_2, width=bar_width, zorder=2, 
                        label='Group 2', **style_2)
        
        data_artists = list(rects1) + list(rects2)
        
        # CRITICAL: Store complete metadata for BOTH axis groups
        bar_info_list_1, bar_info_list_2 = [], []
        
        for i, r in enumerate(rects1):
            bar_info_list_1.append({
                'center': r.get_x() + r.get_width()/2, 
                'height': r.get_height(), 
                'width': r.get_width(),
                'bottom': r.get_y(),
                'top': r.get_y() + r.get_height(),
                'axis': 'primary'
            })
        
        for i, r in enumerate(rects2):
            bar_info_list_2.append({
                'center': r.get_x() + r.get_width()/2, 
                'height': r.get_height(), 
                'width': r.get_width(),
                'bottom': r.get_y(),
                'top': r.get_y() + r.get_height(),
                'axis': 'secondary'
            })
        
        error_artists_1, error_tops_1 = add_error_bars(ax, bar_info_list_1, orientation='vertical')
        error_artists_2, error_tops_2 = add_error_bars(ax2, bar_info_list_2, orientation='vertical')
        
        other_artists.extend(error_artists_1)
        other_artists.extend(error_artists_2)
        
        combined_error_tops = error_tops_1 + error_tops_2
        
        ax.set_xlabel(random.choice(SCIENTIFIC_X_LABELS))
        ax.set_ylabel(random.choice(SCIENTIFIC_Y_LABELS))
        ax2.set_ylabel(random.choice(SCIENTIFIC_Y_LABELS))
        
        # CRITICAL: Atomic position and label setting
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        
        ax.set_ylim(0, max_scale_1 * 1.2)
        ax2.set_ylim(0, max_scale_2 * 1.2)
        
        ax.grid(False)
        ax2.grid(False)
        
        if random.random() < 0.05:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', length=0)
        
        scale_axis_info = {'primary_scale_axis': 'y', 'secondary_scale_axis': 'y2'}
        
        if debug_mode:
            print(f"DEBUG: Dual-axis chart completed - {len(data_artists)} data artists, {len(other_artists)} other artists")
        
        return data_artists, other_artists, bar_info_list_1 + bar_info_list_2, \
               orientation, combined_error_tops, axis_related_artists, scale_axis_info
    
    # --- STANDARD BAR CHART LOGIC ---
    is_scientific = style_config.get('is_scientific', False)
    style = style_config['style']
    pattern = style_config['pattern']
    
    HATCHES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    
    has_treatment_axis = False
    
    num_bars = random.randint(3, 8)
    max_scale = random.choice([50, 100, 200, 500, 750, 1000, 2000])
    allow_negative = random.random() < 0.15 and not is_scientific
    data_pattern_type = 'diverging' if allow_negative else None
    
    orientation = 'horizontal' if num_bars > 6 and random.random() < 0.40 else 'vertical'
    style_config['orientation'] = orientation
    
    if debug_mode:
        print(f"DEBUG: Standard bar chart - Style: {style}, Pattern: {pattern}, Scientific: {is_scientific}")
        print(f"DEBUG: Standard bar chart - Num bars: {num_bars}, Max scale: {max_scale}, Orientation: {orientation}")
    
    ticks_setter = ax.set_xticks if orientation == 'vertical' else ax.set_yticks
    
    if is_scientific:
        num_groups, bars_per_group = random.randint(2, 6), random.randint(1, 4)
        bar_styles = [{'facecolor': 'white', 'edgecolor': 'black', 'hatch': h} 
                     for h in ['', '////', '....', 'xxxx']]
        random.shuffle(bar_styles)
        
        bar_width = 0.8 / bars_per_group; group_width = bar_width * bars_per_group
        
        for i in range(num_groups):
            group_center = i * (group_width + 0.4)
            data = generate_realistic_data(bars_per_group, max_scale, 
                                         allow_negative=False, domain='scientific')
            
            for j in range(bars_per_group):
                pos = group_center - (group_width / 2) + (j + 0.5) * bar_width
                value = data[j]
                bar_style = bar_styles[j % len(bar_styles)]
                
                if orientation == 'vertical':
                    bar_container = ax.bar(pos, value, width=bar_width, **bar_style, zorder=2)
                    bar_info_list.append({
                        'center': pos, 
                        'height': value, 
                        'width': bar_width,
                        'bottom': 0,
                        'top': value
                    })
                else:
                    bar_container = ax.barh(pos, value, height=bar_width, **bar_style, zorder=2)
                    bar_info_list.append({
                        'center': pos, 
                        'height': value, 
                        'width': bar_width,
                        'bottom': 0,
                        'top': value
                    })
                
                data_artists.extend(bar_container.patches)
        
        ticks_positions = [i * (group_width + 0.4) for i in range(num_groups)]
        tick_labels = [f'Group {g+1}' for g in range(num_groups)]
        
        ticks_setter(ticks_positions, tick_labels)
        
        if len(bar_info_list) == 4 and random.random() < 0.30 and orientation == 'vertical':
            axis_related_artists.extend(add_treatment_key_xaxis(ax, bar_info_list))
            has_treatment_axis = True
    
    else:  # Standard Styles
        palette_name = theme_config.get('palette', 'viridis')
        
        if isinstance(palette_name, list): 
            colors = [palette_name[i % len(palette_name)] for i in range(num_bars * 2)]
        else: 
            cmap = colormaps.get(palette_name); 
            colors = [cmap(i / (num_bars * 1.5)) for i in range(num_bars * 2)]
        
        categories = [f'Category {i+1}' for i in range(num_bars)]
        
        if random.random() < 0.2: 
            categories = [c[:random.randint(5,8)] + '...' if len(c) > 8 else c for c in categories]
        
        if style == 'side_by_side':
            y_values1 = generate_realistic_data(num_bars, max_scale, allow_negative, data_pattern_type)
            y_values2 = generate_realistic_data(num_bars, max_scale, allow_negative, data_pattern_type)
            
            bar_width = 0.35; indices = np.arange(num_bars)
            
            if orientation == 'vertical':
                rects1 = ax.bar(indices - bar_width/2, y_values1, width=bar_width, 
                               label='Series 1', color=colors[0], zorder=3)
                rects2 = ax.bar(indices + bar_width/2, y_values2, width=bar_width, 
                               label='Series 2', color=colors[1], zorder=3)
                
                # CRITICAL: Store metadata for BOTH series
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': indices[i] - bar_width/2, 
                        'height': y_values1[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values1[i],
                        'series_idx': 0
                    })
                    bar_info_list.append({
                        'center': indices[i] + bar_width/2, 
                        'height': y_values2[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values2[i],
                        'series_idx': 1
                    })
            else:
                pos1 = indices - bar_width/2
                pos2 = indices + bar_width/2
                rects1 = ax.barh(pos1, y_values1, height=bar_width, 
                                label='Series 1', color=colors[0], zorder=3)
                rects2 = ax.barh(pos2, y_values2, height=bar_width, 
                                label='Series 2', color=colors[1], zorder=3)
                
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': pos1[i], 
                        'height': y_values1[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values1[i],
                        'series_idx': 0
                    })
                    bar_info_list.append({
                        'center': pos2[i], 
                        'height': y_values2[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values2[i],
                        'series_idx': 1
                    })
            
            ticks_setter(indices, categories)
            data_artists.extend(list(rects1) + list(rects2))
        
        elif style == 'stacked':
            y_values1 = generate_realistic_data(num_bars, max_scale/2, allow_negative=False)
            y_values2 = generate_realistic_data(num_bars, max_scale/2, allow_negative=False)
            
            bar_width = 0.7; indices = np.arange(num_bars)
            
            if orientation == 'vertical':
                rects1 = ax.bar(indices, y_values1, width=bar_width, 
                               label='Portion 1', color=colors[0], zorder=3)
                rects2 = ax.bar(indices, y_values2, width=bar_width, bottom=y_values1,
                               label='Portion 2', color=colors[1], zorder=3)
            else:
                rects1 = ax.barh(indices, y_values1, height=bar_width, 
                                label='Portion 1', color=colors[0], zorder=3)
                rects2 = ax.barh(indices, y_values2, height=bar_width, left=y_values1,
                                label='Portion 2', color=colors[1], zorder=3)
            
            ticks_setter(indices, categories)
            data_artists.extend(list(rects1) + list(rects2))
            
            # CRITICAL: Store metadata for EACH stacked segment
            for i in range(num_bars):
                center_pos = indices[i]
                # Bottom segment
                bar_info_list.append({
                    'center': center_pos, 
                    'height': y_values1[i], 
                    'width': bar_width,
                    'bottom': 0,
                    'top': y_values1[i],
                    'series_idx': 0,
                    'bar_idx': i
                })
                # Top segment
                bar_info_list.append({
                    'center': center_pos, 
                    'height': y_values2[i], 
                    'width': bar_width,
                    'bottom': y_values1[i],  # CRITICAL: Bottom of top segment
                    'top': y_values1[i] + y_values2[i],  # CRITICAL: Cumulative top
                    'series_idx': 1,
                    'bar_idx': i
                })
        
        else:  # default, touching, 3d_effect
            y_values = generate_realistic_data(num_bars, max_scale, allow_negative, data_pattern_type)
            
            bar_width = 0.95 if style == 'touching' else 0.8
            gap = 0.05 if style == 'touching' else 0.2
            indices = np.arange(num_bars) * (bar_width + gap)
            
            if orientation == 'vertical':
                rects = ax.bar(indices, y_values, width=bar_width, color=colors, zorder=3)
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': indices[i], 
                        'height': y_values[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values[i]
                    })
            else:
                rects = ax.barh(indices, y_values, height=bar_width, color=colors, zorder=3)
                for i in range(num_bars):
                    bar_info_list.append({
                        'center': indices[i], 
                        'height': y_values[i], 
                        'width': bar_width,
                        'bottom': 0,
                        'top': y_values[i]
                    })
            
            ticks_setter(indices, categories)
            data_artists.extend(list(rects))
    
    # Apply pattern styles
    if pattern != 'none':
        for bar in data_artists:
            fc = bar.get_facecolor()
            if pattern == 'hollow': 
                bar.set_facecolor('none'); bar.set_edgecolor(fc); bar.set_linewidth(1.5)
            elif pattern == 'dotted': 
                bar.set_hatch('..')
            elif pattern == 'striped': 
                bar.set_hatch('//')
            elif pattern == 'hatch': 
                bar.set_hatch(random.choice(HATCHES))
    
    if style == '3d_effect': 
        add_bar_shadows(ax, data_artists, ax.figure)
    
    if not is_scientific and random.random() < 0.3 and style != 'stacked':
        add_jitter_overlay(ax, bar_info_list, orientation)
    
    # --- Coordinated logic for error bars and significance markers ---
    error_bar_artists, error_tops = [], []
    
    if (is_scientific or (not is_scientific and random.random() < 0.30)) and style != 'stacked':
        error_bar_artists, error_tops = add_error_bars(ax, bar_info_list, orientation)
        other_artists.extend(error_bar_artists)
    else:
        error_tops = [b['height'] for b in bar_info_list]
    
    if random.random() < 0.1 and style != 'stacked':
        data_label_artists = add_data_labels(ax, data_artists, orientation, 'bar', 
                                            error_tops=error_tops, bar_info_list=bar_info_list)
        other_artists.extend(data_label_artists)
    
    if is_scientific:
        y_max_limit = ax.get_ylim()[1] if orientation == 'vertical' else ax.get_xlim()[1]
        other_artists.extend(add_significance_markers(ax, bar_info_list, y_max_limit, 
                                                     orientation, error_tops=error_tops))
    
    if allow_negative:
        if orientation == 'vertical': 
            ax.axhline(0, color='black', linewidth=0.8, zorder=1)
        else: 
            ax.axvline(0, color='black', linewidth=0.8, zorder=1)
    else:
        if orientation == 'vertical': 
            ax.set_ylim(bottom=0)
        else: 
            ax.set_xlim(left=0)
    
    ax.set_ylabel(random.choice(SCIENTIFIC_Y_LABELS if is_scientific else BUSINESS_Y_LABELS))
    
    if not has_treatment_axis:
        ax.set_xlabel(random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
    
    if random.random() < 0.3 and orientation == 'vertical': 
        ax.tick_params(axis='x', labelrotation=0)
    
    if is_scientific and orientation == 'vertical' and random.random() < 0.05:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', length=0)
    
    # Apply theme and typography variation
    theme = apply_chart_theme(ax, theme_name, orientation)
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')
    
    # CRITICAL: Ensure all code paths return complete metadata
    if not bar_info_list:
        # Emergency fallback: extract from data_artists
        for artist in data_artists:
            if isinstance(artist, patches.Rectangle):
                if orientation == 'vertical':
                    bar_info_list.append({
                        'center': artist.get_x() + artist.get_width() / 2,
                        'height': artist.get_height(),
                        'width': artist.get_width(),
                        'bottom': artist.get_y(),
                        'top': artist.get_y() + artist.get_height()
                    })
                else:
                    bar_info_list.append({
                        'center': artist.get_y() + artist.get_height() / 2,
                        'height': artist.get_width(),
                        'width': artist.get_height(),
                        'bottom': artist.get_x(),
                        'top': artist.get_x() + artist.get_width()
                    })
    
    if debug_mode:
        print(f"DEBUG: _generate_bar_chart returning - data_artists: {len(data_artists)}, other_artists: {len(other_artists)}, bar_info_list: {len(bar_info_list)}")
        print(f"DEBUG: Scale axis info: {scale_axis_info}")
    
    return data_artists, other_artists, bar_info_list, orientation, error_tops, \
           axis_related_artists, scale_axis_info, None  # No keypoint data for bar charts

'''
def _generate_line_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """Enhanced line chart with keypoint detection."""
    theme = apply_chart_theme(ax, theme_name)
    
    num_series = random.randint(1, 4)
    num_points = random.randint(8, 25)
    max_scale = random.choice([50, 100, 500, 1000])
    
    data_artists = []
    other_artists = []
    keypoint_info = []
    x = np.arange(num_points)
    
    # ========================================================================
    # FIX: Generate colors as a LIST, not a generator
    # ========================================================================
    palette = theme.get('palette', 'viridis')
    
    if isinstance(palette, list):
        # Case 1: Palette is already a list
        # Extend it cyclically if num_series > len(palette)
        colors = [palette[i % len(palette)] for i in range(num_series)]
    else:
        # Case 2: Palette is a colormap name (string)
        try:
            cmap = colormaps.get(palette)
            # Generate num_series colors from the colormap (normalized 0-1)
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
        except (ValueError, KeyError):
            # Fallback: Use viridis if palette name invalid
            cmap = colormaps.get('viridis')
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
    
    # Verify colors generated correctly
    if debug_mode:
        print(f"DEBUG LINE: Generated {len(colors)} colors for {num_series} series")
        print(f"DEBUG LINE: Palette: {palette}, Type: {type(palette)}")
    
    # Ensure we have at least num_series colors (shouldn't happen, but safety check)
    if len(colors) < num_series:
        # Extend cyclically
        colors = colors * ((num_series // len(colors)) + 1)
    
    colors = colors[:num_series]  # Truncate to exact num_series
    
    for series_idx in range(num_series):
        y_data = generate_realistic_data(num_points, max_scale, allow_negative=is_scientific,
                                        domain='scientific' if is_scientific else 'business')
        
        line, = ax.plot(x, y_data, marker='o', markersize=5, linewidth=2,
                       color=colors[series_idx], label=f'Series {series_idx+1}', zorder=3)
        data_artists.append(line)
        
        # After line generation, ensure data coordinates are valid
        y_data_validated = np.clip(y_data, -max_scale * 1.5, max_scale * 1.5)

        inflection_pts = detect_inflection_points(x, y_data_validated, threshold=0.1)
        # Adaptive prominence: higher for noisy data
        prominence_factor = 0.08 if is_scientific else 0.05
        peaks, valleys = detect_extrema(x, y_data_validated, prominence_factor=prominence_factor)

        keypoint_info.append({
            'series_idx': series_idx,
            'start': (float(x[0]), float(y_data_validated[0]), 0),
            'end': (float(x[-1]), float(y_data_validated[-1]), len(x)-1),
            'inflections': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in inflection_pts],
            'peaks': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in peaks],
            'valleys': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in valleys],
            'boundary_points': [(float(x[i]), float(y_data_validated[i]), int(i)) for i in range(len(x))],
            'all_points': [(float(x[i]), float(y_data_validated[i]), int(i)) for i in range(len(x))]
        })
    
    ax.set_xlabel(random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
    ax.set_ylabel(random.choice(SCIENTIFIC_Y_LABELS if is_scientific else BUSINESS_Y_LABELS))
    
    if num_series > 1 and random.random() < 0.7:
        # Substitui a chamada simples por nossa nova função
        legend = apply_legend_variation(ax, num_series)
        other_artists.append(legend)
    
    return data_artists, other_artists, [], 'vertical', None, [], {'primary_scale_axis': 'y'}, keypoint_info
'''

def _generate_line_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """
    CRITICAL FIX: Ensure y_data used for plotting MATCHES y_data used for keypoint annotation.
    The bug was: plot(x, y_data) but annotate with y_data_validated (created AFTER plot).
    """
    theme = apply_chart_theme(ax, theme_name)
    num_series = random.randint(1, 4)
    num_points = random.randint(8, 25)
    max_scale = random.choice([50, 100, 500, 1000])
    
    data_artists = []
    other_artists = []
    keypoint_info = []
    
    x = np.arange(num_points)
    
    # ========================================================================
    # FIX: Generate colors as a LIST, not a generator
    # ========================================================================
    palette = theme.get('palette', 'viridis')
    
    if isinstance(palette, list):
        # Case 1: Palette is already a list
        # Extend it cyclically if num_series > len(palette)
        colors = [palette[i % len(palette)] for i in range(num_series)]
    else:
        # Case 2: Palette is a colormap name (string)
        try:
            cmap = colormaps.get(palette)
            # Generate num_series colors from the colormap (normalized 0-1)
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
        except (ValueError, KeyError):
            # Fallback: Use viridis if palette name invalid
            cmap = colormaps.get('viridis')
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
    
    # Verify colors generated correctly
    if debug_mode:
        print(f"DEBUG LINE: Generated {len(colors)} colors for {num_series} series")
        print(f"DEBUG LINE: Palette: {palette}, Type: {type(palette)}")
    
    # Ensure we have at least num_series colors (shouldn't happen, but safety check)
    if len(colors) < num_series:
        # Extend cyclically
        colors = colors * ((num_series // len(colors)) + 1)
    
    colors = colors[:num_series]  # Truncate to exact num_series
    
    line_styles = ['-', '--', '-.', ':']
    markers = [None, 'o', '^', 's', 'D', 'v', 'p', '*']
    
    for series_idx in range(num_series):
        # Generate raw data
        y_data_raw = generate_realistic_data(num_points, max_scale, allow_negative=is_scientific,
                                        domain='scientific' if is_scientific else 'business')
        
        # CRITICAL FIX: Validate and clip data BEFORE plotting and annotation
        # This ensures the plotted data and annotated data are IDENTICAL
        y_data = np.clip(y_data_raw, -max_scale * 1.5, max_scale * 1.5)
        
        # Ensure data is finite (no NaN or inf)
        y_data = np.nan_to_num(y_data, nan=0.0, posinf=max_scale, neginf=-max_scale)
        
        if debug_mode:
            print(f"DEBUG [LINE] Series {series_idx}: Generated {len(y_data)} points, range [{np.min(y_data):.2f}, {np.max(y_data):.2f}]")
        
        linestyle = random.choice(line_styles)
        marker = random.choice(markers)
        linewidth = random.uniform(1.5, 3.0)
        
        # Plot with the VALIDATED data
        line, = ax.plot(x, y_data, marker=marker, markersize=6 if marker else 0, linewidth=linewidth,
                       linestyle=linestyle, color=colors[series_idx], label=f'Series {series_idx+1}', zorder=3)
        data_artists.append(line)
        
        # Detect keypoints using the SAME validated data
        inflection_pts = detect_inflection_points(x, y_data, threshold=0.1)
        # Adaptive prominence: higher for noisy data
        prominence_factor = 0.08 if is_scientific else 0.05
        peaks, valleys = detect_extrema(x, y_data, prominence_factor=prominence_factor)
        
        # CRITICAL FIX: Capture the exact plotted arrays after plotting to build pose keypoints
        # Use only the coordinates actually plotted for this series to build pose keypoints
        plotted = [(float(x[i]), float(y_data[i]), int(i)) for i in range(len(y_data))]
        series_info = {"plotted_points": plotted}
        
        if debug_mode:
            print(f"DEBUG [LINE] Series {series_idx}: Detected {len(inflection_pts)} inflection points, {len(peaks)} peaks, {len(valleys)} valleys")
            # Log coordinate values in data space 
            print(f"DEBUG [LINE] Series {series_idx}: Data space coords - x: [{x[0]:.2f}, {x[-1]:.2f}], y: [{np.min(y_data):.2f}, {np.max(y_data):.2f}]")
            if len(inflection_pts) > 0:
                print(f"DEBUG [LINE] Series {series_idx}: First inflection - x: {inflection_pts[0][0]:.2f}, y: {inflection_pts[0][1]:.2f}")
            if len(peaks) > 0:
                print(f"DEBUG [LINE] Series {series_idx}: First peak - x: {peaks[0][0]:.2f}, y: {peaks[0][1]:.2f}")
            if len(valleys) > 0:
                print(f"DEBUG [LINE] Series {series_idx}: First valley - x: {valleys[0][0]:.2f}, y: {valleys[0][1]:.2f}")
            print(f"DEBUG [LINE] Series {series_idx}: Captured {len(plotted)} plotted points for pose construction")
        
        # CRITICAL FIX: Store keypoints with explicit float conversion and validation
        keypoint_info.append({
            'series_idx': series_idx,
            'start': (float(x[0]), float(y_data[0]), 0),
            'end': (float(x[-1]), float(y_data[-1]), len(x)-1),
            'inflections': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in inflection_pts],
            'peaks': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in peaks],
            'valleys': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in valleys],
            'boundary_points': [(float(x[i]), float(y_data[i]), int(i)) for i in range(len(x))],
            'all_points': [(float(x[i]), float(y_data[i]), int(i)) for i in range(len(x))],
            'plotted_points': plotted  # Add the plotted points for pose construction
        })
        
        if debug_mode:
            print(f"DEBUG [LINE] Series {series_idx}: Keypoint info stored - start={keypoint_info[-1]['start']}, end={keypoint_info[-1]['end']}")
            # Log all_points for verification
            if keypoint_info[-1]['all_points']:
                print(f"DEBUG [LINE] Series {series_idx}: First point: ({keypoint_info[-1]['all_points'][0][0]:.2f}, {keypoint_info[-1]['all_points'][0][1]:.2f}), Last point: ({keypoint_info[-1]['all_points'][-1][0]:.2f}, {keypoint_info[-1]['all_points'][-1][1]:.2f})")
    
    ax.set_xlabel(random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
    ax.set_ylabel(random.choice(SCIENTIFIC_Y_LABELS if is_scientific else BUSINESS_Y_LABELS))
    
    if num_series > 1 and random.random() < 0.7:
        # Substitui a chamada simples por nossa nova função
        legend = apply_legend_variation(ax, num_series)
        other_artists.append(legend)
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')
    
    # --- INÍCIO DA MODIFICAÇÃO ---
    # Coleta o valor y mínimo de todas as séries para verificar a escala de log
    all_y_vals = [pt[1] for kpi in keypoint_info for pt in kpi['plotted_points']]
    data_min = np.min(all_y_vals) if all_y_vals else 0
    apply_axis_scaling(ax, data_min=data_min, orientation='vertical')
    # --- FIM DA MODIFICAÇÃO ---
    
    return data_artists, other_artists, [], 'vertical', None, [], {'primary_scale_axis': 'y'}, keypoint_info


def _generate_scatter_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """Enhanced scatter with realistic correlation structures and sample sizes"""
    theme = apply_chart_theme(ax, theme_name)
    palette = theme.get('palette', 'viridis')
    
    # Realistic sample sizes based on publication analysis
    if is_scientific:
        num_points = np.random.choice([15, 20, 25, 30, 50, 75, 100], 
                                     p=[0.15, 0.25, 0.20, 0.15, 0.15, 0.05, 0.05])
    else:
        num_points = np.random.choice([50, 100, 200, 500], 
                                     p=[0.20, 0.40, 0.30, 0.10])
    
    max_scale = random.choice([50, 100, 200, 500, 1000])
    
    # Realistic correlation patterns with realistic R² values
    relationship = random.choices([
        'strong_positive',    # R² = 0.64-0.81
        'moderate_positive',  # R² = 0.36-0.64  
        'weak_positive',      # R² = 0.09-0.36
        'no_correlation',     # R² = 0-0.09
        'strong_negative',    # R² = 0.64-0.81
        'moderate_negative',  # R² = 0.36-0.64
        'nonlinear',         # Quadratic, exponential relationships
        'clustered'          # Multiple distinct groups
    ], weights=[0.20, 0.25, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05])[0]
    
    data_artists = []
    other_artists = []
    
    # Generate X data with realistic distribution
    x_data = generate_realistic_data(
        num_points, max_scale, 
        allow_negative=False,
        pattern_type='linear' if relationship != 'clustered' else None,
        domain='scientific' if is_scientific else 'business'
    )
    
    if relationship == 'clustered':
        # Generate realistic cluster data
        num_clusters = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
        x_data, y_data = [], []
        
        for cluster_idx in range(num_clusters):
            cluster_size = max(5, num_points // num_clusters + np.random.randint(-3, 4))
            
            # Well-separated cluster centers
            mean_x = random.uniform(0.15, 0.85) * max_scale
            mean_y = random.uniform(0.15, 0.85) * max_scale
            
            # Realistic covariance (elliptical clusters)
            cov_xx = random.uniform(0.015, 0.08) * max_scale**2
            cov_yy = random.uniform(0.015, 0.08) * max_scale**2
            cov_xy = random.uniform(-0.5, 0.5) * np.sqrt(cov_xx * cov_yy)
            
            cov = [[cov_xx, cov_xy], [cov_xy, cov_yy]]
            cluster_data = np.random.multivariate_normal([mean_x, mean_y], cov, cluster_size)
            
            x_data.extend(cluster_data[:, 0])
            y_data.extend(cluster_data[:, 1])
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
    else:
        # Sort X for clearer trend visualization
        x_data = np.sort(x_data)
        
        # Generate Y based on X with realistic correlations
        if relationship in ['strong_positive', 'strong_negative']:
            # R² ≈ 0.75-0.90
            target_r_squared = np.random.uniform(0.64, 0.81)
            slope = np.random.uniform(0.5, 1.5) * (1 if 'positive' in relationship else -1)
            
        elif relationship in ['moderate_positive', 'moderate_negative']:
            # R² ≈ 0.40-0.65
            target_r_squared = np.random.uniform(0.36, 0.64)
            slope = np.random.uniform(0.3, 0.8) * (1 if 'positive' in relationship else -1)
            
        elif relationship in ['weak_positive', 'weak_negative']:
            # R² ≈ 0.10-0.35
            target_r_squared = np.random.uniform(0.09, 0.36)
            slope = np.random.uniform(0.1, 0.5) * (1 if 'positive' in relationship else -1)
            
        elif relationship == 'no_correlation':
            # R² ≈ 0-0.09
            target_r_squared = np.random.uniform(0.0, 0.09)
            slope = np.random.uniform(-0.2, 0.2)
        
        elif relationship == 'nonlinear':
            # Nonlinear relationships
            nonlinear_type = np.random.choice(['quadratic', 'exponential', 'logarithmic'])
            
            if nonlinear_type == 'quadratic':
                a = np.random.uniform(-0.002, 0.002)
                b = np.random.uniform(-0.5, 0.5)
                c = np.random.uniform(0.2, 0.5) * max_scale
                y_data = a * x_data**2 + b * x_data + c
                
            elif nonlinear_type == 'exponential':
                a = np.random.uniform(0.05, 0.15)
                b = np.random.uniform(0.01, 0.03)
                c = np.random.uniform(0, max_scale * 0.2)
                y_data = a * np.exp(b * x_data / max_scale) + c
                y_data = np.clip(y_data, 0, max_scale)
                
            elif nonlinear_type == 'logarithmic':
                a = np.random.uniform(10, 30)
                b = np.random.uniform(0, max_scale * 0.3)
                y_data = a * np.log(x_data + 1) + b
            
            # Add noise
            noise_cv = np.random.uniform(0.08, 0.15)
            noise = np.random.normal(0, np.abs(y_data) * noise_cv, num_points)
            y_data += noise
        else:
            # Default linear
            target_r_squared = 0.5
            slope = 0.5
        
        if relationship != 'nonlinear':
            # Generate linear relationship with specific R²
            intercept = max_scale * np.random.uniform(0.1, 0.3)
            y_perfect = slope * (x_data - np.mean(x_data)) + intercept
            
            # Add noise to achieve target R²
            noise_variance = np.var(y_perfect) * ((1 - target_r_squared) / target_r_squared)
            noise = np.random.normal(0, np.sqrt(noise_variance), num_points)
            y_data = y_perfect + noise
    
    # Ensure realistic bounds
    y_data = np.clip(y_data, 0 if not is_scientific else -max_scale*0.2, max_scale * 1.2)
    x_data = np.clip(x_data, 0, max_scale * 1.2)
    
    # Realistic point styling based on sample size
    scatter_kwargs = {
        'alpha': np.random.uniform(0.6, 0.8),
        'zorder': 2,
        'marker': np.random.choice(['o', 's', '^', 'D', '+'])
    }
    
    # Size scaling with sample size (larger datasets = smaller points)
    if num_points < 30:
        scatter_kwargs['s'] = np.random.randint(60, 100)
    elif num_points < 100:
        scatter_kwargs['s'] = np.random.randint(30, 60)
    elif num_points < 500:
        scatter_kwargs['s'] = np.random.randint(15, 30)
    else:
        scatter_kwargs['s'] = np.random.randint(5, 15)
    
    # Store point size for radius calculation
    point_size = scatter_kwargs['s']
    
    # Color strategy
    if isinstance(palette, list) and palette:
        scatter_kwargs['c'] = palette[0]
    else:
        scatter_kwargs['c'] = '#2E86AB'
    
    scatter = ax.scatter(x_data, y_data, **scatter_kwargs)
    data_artists.append(scatter)
    
    # Add trend line for correlated data
    if relationship not in ['no_correlation', 'clustered'] and np.random.random() < 0.7:
        # Fit trend line
        coeffs = np.polyfit(x_data, y_data, 1)
        trend_line = np.poly1d(coeffs)
        x_trend = np.linspace(np.min(x_data), np.max(x_data), 100)
        
        line_color = palette[1] if isinstance(palette, list) and len(palette) > 1 else '#D62728'
        line, = ax.plot(x_trend, trend_line(x_trend), '--', 
                       color=line_color, alpha=0.8, linewidth=2, zorder=1)
        other_artists.append(line)
    
    # Set labels
    ax.set_xlabel(np.random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
    ax.set_ylabel(np.random.choice(SCIENTIFIC_Y_LABELS if is_scientific else BUSINESS_Y_LABELS))
    
    # Realistic axis limits with padding
    x_range = np.max(x_data) - np.min(x_data)
    y_range = np.max(y_data) - np.min(y_data)
    
    ax.set_xlim(np.min(x_data) - 0.05*x_range, np.max(x_data) + 0.05*x_range)
    ax.set_ylim(np.min(y_data) - 0.05*y_range, np.max(y_data) + 0.05*y_range)
    
    # **NEW: Build scatter point metadata with coordinates and radius**
    # Convert matplotlib scatter point size to radius in data coordinates
    # scatter 's' parameter is in points^2, radius calculation requires axis transformation
    fig = ax.figure
    dpi = fig.dpi if fig else 72.0
    
    # Calculate approximate radius in data coordinates
    # Point size 's' is area in points^2, so radius in points = sqrt(s/pi)
    radius_points = np.sqrt(point_size / np.pi)
    
    # Transform radius from display coordinates to data coordinates
    # Get approximate data-to-display scaling factor from axis limits
    x_display_range = ax.transData.transform([[ax.get_xlim()[1], 0]])[0][0] - \
                      ax.transData.transform([[ax.get_xlim()[0], 0]])[0][0]
    x_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_scale_factor = x_data_range / x_display_range if x_display_range != 0 else 1.0
    
    y_display_range = ax.transData.transform([[0, ax.get_ylim()[1]]])[0][1] - \
                      ax.transData.transform([[0, ax.get_ylim()[0]]])[0][1]
    y_data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_scale_factor = y_data_range / y_display_range if y_display_range != 0 else 1.0
    
    # Average radius in data coordinates (approximate)
    radius_data = radius_points * (x_scale_factor + y_scale_factor) / 2.0
    
    # Build scatter metadata structure
    scatter_metadata = {
        'relationship': relationship,
        'num_points': num_points,
        'point_size': point_size,
        'radius_data': radius_data,  # Approximate radius in data coordinates
        'points': [
            {
                'x': float(x_data[i]),
                'y': float(y_data[i]),
                'index': i,
                'radius': radius_data  # Same radius for all points (uniform size)
            }
            for i in range(len(x_data))
        ]
    }
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')
    
    # --- INÍCIO DA MODIFICAÇÃO ---
    data_min = np.min(y_data)
    apply_axis_scaling(ax, data_min=data_min, orientation='vertical')
    # --- FIM DA MODIFICAÇÃO ---
    
    return data_artists, other_artists, [], 'vertical', [], [], \
           {'primary_scale_axis': 'x', 'secondary_scale_axis': 'y'}, scatter_metadata

def _generate_boxplot_chart(ax, theme_name, theme_config, is_scientific, 
                           box_width=0.6, outlier_style='circle', show_significance=True, debug_mode=False):
    """Generate boxplot with realistic data and complete annotation metadata"""
    
    # Apply theme
    theme = apply_chart_theme(ax, theme_name)
    if not theme:
        theme = {'palette': 'viridis'}
    
    # MODIFICATION: 15% chance for horizontal boxplot with more boxes
    is_horizontal = random.random() < 0.15
    
    # Generate realistic data - more groups for horizontal orientation
    if is_horizontal:
        num_groups = random.randint(6, 12)  # More boxes for horizontal
    else:
        num_groups = random.randint(3, 8)
    
    max_scale = 100
    
    datas = [generate_realistic_data(num_points=random.randint(20, 50), max_scale=max_scale,
                                   allow_negative=is_scientific, pattern_type=None)
             for _ in range(num_groups)]
    
    # Handle empty or invalid data
    if not datas or any(len(d) == 0 for d in datas):
        return [], [], [], 'vertical', [], [], {}, None
    
    # MODIFICATION: Create boxplot with orientation parameter
    bp = ax.boxplot(datas, patch_artist=True, widths=box_width,
                   vert=not is_horizontal,  # CRITICAL: vert=False for horizontal
                   flierprops={'marker': {'circle': 'o', 'star': '*', 'diamond': 'D'}.get(outlier_style, 'o'),
                              'markersize': 5, 'alpha': 0.6})
    
    # Apply styling
    data_artists = _apply_box_styles(bp, theme, is_scientific)
    _apply_line_styles(bp)  # Median, whisker, cap styles
    
    # MODIFICATION: Jitter overlay adapted for orientation
    if is_scientific and random.random() < 0.2:
        for i, d in enumerate(datas):
            if is_horizontal:
                y_coords = np.random.normal(i + 1, 0.04, size=len(d))
                ax.plot(d, y_coords, '.', color='black', alpha=0.3, zorder=10)
            else:
                x_coords = np.random.normal(i + 1, 0.04, size=len(d))
                ax.plot(x_coords, d, '.', color='black', alpha=0.3, zorder=10)
    
    # MODIFICATION: Set labels based on orientation
    if is_horizontal:
        ax.set_xlabel(random.choice(SCIENTIFIC_Y_LABELS if is_scientific else BUSINESS_Y_LABELS))
        ax.set_ylabel(random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
        ax.set_yticklabels([f'G{i+1}' for i in range(num_groups)])
    else:
        ax.set_ylabel(random.choice(SCIENTIFIC_Y_LABELS if is_scientific else BUSINESS_Y_LABELS))
        ax.set_xlabel(random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
        ax.set_xticklabels([f'G{i+1}' for i in range(num_groups)])
    
    # Collect error bar artists (whiskers and caps)
    error_groups = []
    for g in range(num_groups):
        group_artists = [
            bp['whiskers'][2*g], bp['whiskers'][2*g + 1],
            bp['caps'][2*g], bp['caps'][2*g + 1]
        ]
        error_groups.append(group_artists)
    
    # MODIFICATION: Add significance markers with correct orientation
    bar_info_list = []
    sig_artists = []
    orientation_str = 'horizontal' if is_horizontal else 'vertical'
    
    if show_significance and random.random() < 0.5:
        max_extent = 0
        error_tops = []
        
        for g in range(num_groups):
            if is_horizontal:
                # For horizontal: whiskers extend along x-axis
                top = bp['whiskers'][2*g+1].get_xdata()[1]
                max_extent = max(max_extent, top + abs(top) * 0.1)
                error_tops.append(top)
                center = g + 1
                bar_info_list.append({'center': center, 'height': top, 'width': box_width})
            else:
                # For vertical: whiskers extend along y-axis
                top = bp['whiskers'][2*g+1].get_ydata()[1]
                max_extent = max(max_extent, top + abs(top) * 0.1)
                error_tops.append(top)
                center = g + 1
                bar_info_list.append({'center': center, 'height': top, 'width': box_width})
        
        sig_artists = add_significance_markers(ax, bar_info_list, max_extent, orientation_str, error_tops)
    
    # Collect other artists
    other_artists_list = []
    for group in error_groups:
        other_artists_list.extend(group)
    
    if bp['fliers']:
        other_artists_list.extend(bp['fliers'])
    
    if sig_artists:
        other_artists_list.extend(sig_artists)
    
    # MODIFICATION: Extract median line coordinates adapted for orientation
    median_metadata = []
    
    for group_idx, median_line in enumerate(bp['medians']):
        x_coords = median_line.get_xdata()
        y_coords = median_line.get_ydata()
        
        if len(x_coords) >= 2 and len(y_coords) >= 2:
            if is_horizontal:
                # Horizontal boxplot: median is vertical line (x constant, y varies)
                lower_left = {'x': float(x_coords[0]), 'y': float(y_coords[0])}
                upper_right = {'x': float(x_coords[-1]), 'y': float(y_coords[-1])}
                median_value = float(x_coords[0])  # Constant x-coordinate
                center_y = float((y_coords[0] + y_coords[-1]) / 2.0)
                line_length = float(y_coords[-1] - y_coords[0])
                
                median_metadata.append({
                    'group_index': group_idx,
                    'group_label': f'G{group_idx+1}',
                    'median_value': median_value,
                    'lower_left': lower_left,
                    'upper_right': upper_right,
                    'center_y': center_y,
                    'line_length': line_length,
                    'orientation': 'horizontal'
                })
            else:
                # Vertical boxplot: median is horizontal line (y constant, x varies)
                lower_left = {'x': float(x_coords[0]), 'y': float(y_coords[0])}
                upper_right = {'x': float(x_coords[-1]), 'y': float(y_coords[-1])}
                median_value = float(y_coords[0])
                center_x = float((x_coords[0] + x_coords[-1]) / 2.0)
                line_length = float(x_coords[-1] - x_coords[0])
                
                median_metadata.append({
                    'group_index': group_idx,
                    'group_label': f'G{group_idx+1}',
                    'median_value': median_value,
                    'lower_left': lower_left,
                    'upper_right': upper_right,
                    'center_x': center_x,
                    'line_length': line_length,
                    'orientation': 'vertical'
                })
    
    # MODIFICATION: Build complete boxplot metadata with orientation info
    boxplot_metadata = {
        'num_groups': num_groups,
        'box_width': box_width,
        'medians': median_metadata,
        'orientation': orientation_str
    }
    
    # MODIFICATION: Return correct orientation and swapped scale axes for horizontal
    if is_horizontal:
        scale_axis_info = {'primary_scale_axis': 'x', 'boxplot_raw': bp}
    else:
        scale_axis_info = {'primary_scale_axis': 'y', 'boxplot_raw': bp}
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')
    
    return data_artists, other_artists_list, bar_info_list, orientation_str, [], [], \
           scale_axis_info, boxplot_metadata


def _apply_box_styles(bp, theme, is_scientific):
    """Apply styling to boxplot boxes based on theme and context"""
    data_artists = bp['boxes']
    num_groups = len(data_artists)
    
    if is_scientific and random.random() < 0.9:
        scientific_style = random.choice(['hollow', 'grayscale'])
        
        if scientific_style == 'hollow':
            for patch in data_artists:
                patch.set_facecolor('none')
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)
        
        elif scientific_style == 'grayscale':
            colors = ['#FFFFFF', '#DDDDDD', '#BBBBBB', '#999999']
            for i, patch in enumerate(data_artists):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)
    
    else:
        palette = theme.get('palette', 'viridis')
        colors = []
        
        if isinstance(palette, list):
            colors = [palette[i % len(palette)] for i in range(num_groups)]
        elif isinstance(palette, str):
            try:
                cmap = colormaps.get(palette)
                colors = [cmap(i / num_groups) for i in range(num_groups)]
            except ValueError:
                cmap = colormaps.get('viridis')
                colors = [cmap(i / num_groups) for i in range(num_groups)]
        
        for patch, color in zip(data_artists, colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
    
    return data_artists


def _apply_line_styles(bp):
    """Apply consistent styling to medians, whiskers, and caps"""
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)

def _generate_pie_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """Enhanced pie chart with geometric keypoint calculation."""
    theme = apply_chart_theme(ax, theme_name)
    
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    num_slices = random.randint(3, 8)
    data = [random.uniform(10, 100) for _ in range(num_slices)]
    # Gera os nomes dos rótulos de texto
    labels_text = [f'Item {i+1}' for i in range(num_slices)]
    
    explode = [0.0] * num_slices
    if random.random() < 0.4:
        explode[random.randint(0, num_slices-1)] = 0.10
    
    palette = theme.get('palette', 'viridis')
    colors = [palette[i % len(palette)] for i in range(num_slices)] if isinstance(palette, list) else [colormaps.get(palette)(i/num_slices) for i in range(num_slices)]
    
    # --- NOVA LÓGICA DE RÓTULOS ---
    # Chama a nova função ANTES de plotar
    pie_params = apply_pie_label_strategy(data, labels_text)
    
    # --- START OF FIX ---
    # Handle variable return values from ax.pie()
    
    autotexts = [] # Initialize autotexts as an empty list
    
    if pie_params.get('autopct') is not None:
        # autopct is provided, so expect 3 return values
        wedges, texts, autotexts = ax.pie(data, explode=explode, colors=colors,
                                           startangle=90,
                                           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                                           **pie_params)
    else:
        # autopct is None, so expect 2 return values
        wedges, texts = ax.pie(data, explode=explode, colors=colors,
                                 startangle=90,
                                 wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                                 **pie_params)
    # --- END OF FIX ---
    ax.axis('equal')

    if debug_mode:
        print(f"DEBUG [PIE] Generated {len(wedges)} wedges with {num_slices} slices")
        print(f"DEBUG [PIE] Data values: {[f'{d:.1f}' for d in data]}")

    pie_geometry = calculate_pie_geometry(wedges, ax, debug_mode)

    if debug_mode and pie_geometry:
        print(f"DEBUG [PIE] Pie geometry calculated - center: {pie_geometry.get('center_point')}")
        print(f"DEBUG [PIE] Number of wedges in geometry: {len(pie_geometry.get('wedges', []))}")
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')
    
    # The return value correctly handles autotexts being an empty list
    return wedges, autotexts + texts, [], 'vertical', [], [], {'primary_scale_axis': 'none'}, pie_geometry


def _generate_histogram(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """Generate histogram with realistic data distribution"""
    # Apply general theme settings (background, grid, etc.)
    theme = apply_chart_theme(ax, theme_name)
    
    num_bins = random.randint(8, 20)
    data = np.random.normal(loc=random.uniform(-10, 10), scale=random.uniform(5, 15), size=500)
    
    # Prioritize using theme's color palette
    hist_color = None
    palette = theme.get('palette')
    
    if isinstance(palette, list) and palette:
        # If palette is a list of colors, choose one
        hist_color = random.choice(palette)
    elif isinstance(palette, str):
        # If palette is a colormap name, get a representative color from it
        try:
            # Get a color from the middle of the colormap
            hist_color = colormaps.get(palette)(0.4)
        except (ValueError, AttributeError):
            # If colormap name is invalid, hist_color remains None
            pass
    
    # Fall back to original hardcoded colors if theme palette is not usable
    if hist_color is None:
        hist_color = random.choice(['#4472C4', '#5B9BD5', '#66C2A5'])
    
    n, bins, patches = ax.hist(data, bins=num_bins, color=hist_color, zorder=2)
    
    data_artists = patches
    
    bar_info_list = []
    for r in data_artists:
        bar_info_list.append({
            'center': r.get_x() + r.get_width()/2, 
            'height': r.get_height(), 
            'width': r.get_width(),
            'bottom': 0,
            'top': r.get_height()
        })
    
    ax.set_ylabel(random.choice(HISTOGRAM_Y_LABELS))
    ax.set_xlabel(random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
    
    # Histogram data labels typically show frequency/count values on top of bars
    data_label_artists = []
    
    # Add data labels with 10% probability (histograms don't always have labels)
    if random.random() < 0.1:
        # Select subset of bars to label (not all bars, typically higher frequency ones)
        # Sort bars by height and label top 30-50% of bars
        sorted_bars = sorted(zip(patches, n), key=lambda x: x[1], reverse=True)
        num_to_label = max(3, int(len(sorted_bars) * random.uniform(0.3, 0.5)))
        bars_to_label = sorted_bars[:num_to_label]
        
        for patch, height in bars_to_label:
            if height > 0:  # Only label non-empty bars
                x_pos = patch.get_x() + patch.get_width() / 2.0
                y_pos = height
                
                # Format label: show integer count
                label_text = f'{int(height)}'
                
                # Add text annotation
                text_artist = ax.text(
                    x_pos, y_pos,
                    label_text,
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black',
                    zorder=3
                )
                data_label_artists.append(text_artist)
    
    # Apply typography variation
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')
    
    # Return 8 values: include data_label_artists in other_artists
    # Data labels are "other_artists" not "data_artists"
    return data_artists, data_label_artists, bar_info_list, 'vertical', [], [], \
           {'primary_scale_axis': 'y'}, None

def _generate_area_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """
    CRITICAL FIX: Garante que boundary_y usado para plotar CORRESPONDE a boundary_y usado para anotação.
    NOVO: Adiciona 'stacking_mode' para 'stacked', 'overlapping', ou 'percentage'.
    CORREÇÃO (Usuário): Garante que os dados da série não excedam a escala da série individual,
                     evitando que a soma empilhada exceda o max_scale total.
    CORREÇÃO (Definitiva 2): Usa o MÁXIMO DE DADOS REAIS (y_stack ou all_series_data)
                             para definir o ylim, em vez do max_scale teórico.
    """
    theme = apply_chart_theme(ax, theme_name)
    num_series = random.randint(1, 4)
    num_points = random.randint(8, 25)
    max_scale = random.choice([50, 100, 500, 1000])
    
    # --- NOVA LÓGICA DE MODO DE EMPILHAMENTO ---
    stacking_mode = random.choice(['stacked', 'overlapping', 'percentage'])
    if debug_mode:
        print(f"DEBUG [AREA] Stacking Mode: {stacking_mode}")

    data_artists = []
    other_artists = []
    keypoint_info = []
    x = np.arange(num_points)
    
    # ========================================================================
    # FIX: Gerar cores como uma LISTA
    # ========================================================================
    palette = theme.get('palette', 'viridis')
    
    if isinstance(palette, list):
        colors = [palette[i % len(palette)] for i in range(num_series)]
    else:
        try:
            cmap = colormaps.get(palette)
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
        except (ValueError, KeyError):
            cmap = colormaps.get('viridis')
            colors = [cmap(i / max(1, num_series - 1)) for i in range(num_series)]
    
    if debug_mode:
        print(f"DEBUG AREA: Generated {len(colors)} colors for {num_series} series")
    
    if len(colors) < num_series:
        colors = colors * ((num_series // len(colors)) + 1)
    
    colors = colors[:num_series]
    
    # --- 1. GERAR TODOS OS DADOS PRIMEIRO ---
    all_series_data = []  # Ensure initialized
    
    # Determine per-series max_scale based on stacking mode
    if stacking_mode == 'stacked':
        # For stacked: each series max ~ max_scale / num_series to keep total ~ max_scale
        series_max = max_scale / max(1, num_series)
    elif stacking_mode == 'overlapping':
        # For overlapping: each series max ~ max_scale (independent)
        series_max = max_scale
    else:  # 'percentage'
        # For percentage: generate with max_scale, will normalize later
        series_max = max_scale

    if debug_mode:
        print(f"DEBUG AREA: Stacking mode={stacking_mode}, Total max_scale={max_scale}")
        print(f"DEBUG AREA: Num series={num_series}, Per-series max={series_max:.2f}")

    for series_idx in range(num_series):
        # Generate raw data with appropriate per-series scale
        data_raw = generate_realistic_data(num_points, series_max, 
                                          allow_negative=False,
                                          domain='scientific' if is_scientific else 'business')
        
        if stacking_mode == 'percentage':
            # For percentage mode, generate positive values only (will normalize later)
            data = np.maximum(0, data_raw)
        else:
            # For stacked/overlapping, allow realistic values but clip to per-series max
            data = np.clip(np.maximum(0, data_raw), 0, series_max * 1.1)
        
        all_series_data.append(data.copy())
        
        if debug_mode:
            print(f"DEBUG AREA: Series {series_idx} data range: {np.min(data):.2f} to {np.max(data):.2f}")

    # For percentage mode: Normalize after all series generated
    if stacking_mode == 'percentage':
        total_sum_y = np.sum(all_series_data, axis=0)  # Sum across series at each x-point
        total_sum_y = np.maximum(total_sum_y, 1e-6)  # Avoid division by zero
        
        normalized_data = []
        for data in all_series_data:
            # Normalize each series to percentage of total at each x-point
            normalized = (data / total_sum_y) * 100.0
            normalized_data.append(normalized)
        
        all_series_data = normalized_data
        
        if debug_mode:
            print(f"DEBUG AREA: Percentage mode - Normalized data sums to 100%")
            # Verify normalization
            for i, data in enumerate(all_series_data):
                total_at_x = np.sum([d[i] for d in all_series_data], axis=0)
                print(f"DEBUG AREA: Series {i} normalization check - max sum: {np.max(total_at_x):.2f}")



    # --- 3. LOOP DE PLOTAGEM E ANOTAÇÃO ---
    y_stack = np.zeros(num_points) # Base para 'stacked' e 'percentage'

    for series_idx, data in enumerate(all_series_data):
        color = colors[series_idx]
        
        # --- LÓGICA DE EMPILHAMENTO ---
        if stacking_mode == 'overlapping':
            y_stack_previous = np.zeros(num_points) 
            boundary_y = data 
            alpha = 0.5 
        else: # 'stacked' or 'percentage'
            y_stack_previous = y_stack
            boundary_y = y_stack + data
            alpha = 0.7 
        
        if debug_mode:
            print(f"DEBUG [AREA] Series {series_idx}: y_stack_previous range [{np.min(y_stack_previous):.2f}, {np.max(y_stack_previous):.2f}]")
            print(f"DEBUG [AREA] Series {series_idx}: boundary_y range [{np.min(boundary_y):.2f}, {np.max(boundary_y):.2f}]")
        
        plotted = [(float(x[i]), float(boundary_y[i]), int(i)) for i in range(len(boundary_y))]

        area = ax.fill_between(x, y_stack_previous, boundary_y, color=color, alpha=alpha,
                               label=f'Series {series_idx+1}', zorder=2)
        data_artists.append(area)
        
        line, = ax.plot(x, boundary_y, color='white', linewidth=1.5, alpha=0.9, zorder=3)
        other_artists.append(line)
        
        inflection_pts = detect_inflection_points(x, boundary_y, threshold=0.1)
        prominence_factor = 0.08 if is_scientific else 0.05
        peaks, valleys = detect_extrema(x, boundary_y, prominence_factor=prominence_factor)
        
        keypoint_info.append({
            'series_idx': series_idx,
            'start': (float(x[0]), float(boundary_y[0]), 0),
            'end': (float(x[-1]), float(boundary_y[-1]), len(x)-1),
            'inflections': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in inflection_pts],
            'peaks': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in peaks],
            'valleys': [(float(x_val), float(y_val), int(idx)) for x_val, y_val, idx in valleys],
            'boundary_points': [(float(x[i]), float(boundary_y[i]), int(i)) for i in range(len(x))],
            'fill_bottom': [(float(x[i]), float(y_stack_previous[i]), int(i)) for i in range(len(x))], 
            'fill_top': [(float(x[i]), float(boundary_y[i]), int(i)) for i in range(len(x))],
            'plotted_points': plotted
        })
        
        if stacking_mode != 'overlapping':
            y_stack += data
    
    # --- 4. CONFIGURAÇÃO FINAL DO EIXO ---
    ax.set_xlabel(random.choice(SCIENTIFIC_X_LABELS if is_scientific else BUSINESS_X_LABELS))
    
    if stacking_mode == 'percentage':
        ax.set_ylabel("Percentage (%)")
    else:
        ax.set_ylabel(random.choice(SCIENTIFIC_Y_LABELS if is_scientific else BUSINESS_Y_LABELS))
    
    if num_series > 1 and random.random() < 0.7:
        legend = apply_legend_variation(ax, num_series)
        other_artists.append(legend)
    
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')
    
    # --- INÍCIO DA CORREÇÃO DEFINITIVA (SUA LÓGICA) ---
    
    # 1. Aplicar a escala (log/symlog/linear) PRIMEIRO
    if stacking_mode != 'percentage':
        apply_axis_scaling(ax, data_min=0.01, orientation='vertical')
    
    # 2. Obter a escala que foi definida (pode ser 'log', 'symlog', ou 'linear')
    current_scale = ax.get_yscale()

    # 3. AGORA, definir os limites (ylim)
    if stacking_mode == 'percentage':
        ax.set_ylim(0, 100)
    else:
        # (Minha Lógica) Definir o limite inferior com base na escala
        bottom_limit = None # Deixa o Matplotlib decidir o 'bottom' se for log
        if current_scale != 'log':
            bottom_limit = 0 # Define o 'bottom' como 0 para 'linear' e 'symlog'
        
        # (Sua Lógica) Calcular o limite superior com base nos DADOS REAIS
        if stacking_mode == 'stacked':
            # Para empilhado: usar o máximo do y_stack cumulativo
            actual_max = np.max(y_stack)
        elif stacking_mode == 'overlapping':
            # Para sobreposto: usar o máximo de todas as séries individuais
            actual_max = max(np.max(series) for series in all_series_data)
        
        # Definir limite superior com 15% de preenchimento
        top_limit = actual_max * 1.15 
        
        # Verificação de segurança
        if not np.isfinite(top_limit) or top_limit <= 0:
            top_limit = max_scale * 1.2  # Fallback
        
        # Aplicar os limites calculados
        ax.set_ylim(bottom=bottom_limit, top=top_limit)
        
        if debug_mode:
            print(f"DEBUG [AREA_YLIM] Mode={stacking_mode}, Scale={current_scale}, ActualMax={actual_max:.2f}, TopLimit={top_limit:.2f}, BottomLimit={bottom_limit}")
            
    # --- FIM DA CORREÇÃO DEFINITIVA ---
    
    return data_artists, other_artists, [], 'vertical', None, [], {'primary_scale_axis': 'y'}, keypoint_info

def _generate_heatmap_chart(ax, theme_name, theme_config, is_scientific, debug_mode=False):
    """
    Gera um heatmap com estruturas de dados realistas (correlação, cluster, etc.)
    e usa pcolormesh para anotação robusta de células.
    """
    theme = apply_chart_theme(ax, theme_name)
    apply_typography_variation(ax, domain='scientific' if is_scientific else 'business')

    # 1. Gerar dados estruturados
    data, cmap_type = generate_structured_heatmap(debug_mode=debug_mode)
    rows, cols = data.shape

    # 2. Selecionar colormap apropriado com base no tipo de dados
    COLORMAP_CATEGORIES = {
        'sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                       'Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
                       'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu'],

        'diverging': ['coolwarm', 'bwr', 'seismic', 'RdBu', 'RdGy',
                      'RdYlBu', 'RdYlGn', 'Spectral', 'PiYG', 'PRGn']
    }

    if cmap_type == 'diverging':
        # For diverging data (e.g., correlation matrices), use diverging colormaps
        cmap_name = np.random.choice(COLORMAP_CATEGORIES['diverging'])
    else:  # sequential
        # For sequential data, use sequential colormaps
        cmap_name = np.random.choice(COLORMAP_CATEGORIES['sequential'])

    # 3. Usar pcolormesh (gera QuadMesh, que generator.py lida bem)
    x = np.arange(cols + 1)
    y = np.arange(rows + 1)

    mesh = ax.pcolormesh(x, y, data, cmap=cmap_name, zorder=2,
                           edgecolors='white', linewidth=0.5 if rows*cols < 200 else 0)
    data_artists = [mesh]

    # 4. Adicionar colorbar
    cbar = ax.figure.colorbar(mesh, ax=ax)
    # A lógica de anotação encontra isso
    other_artists = [cbar]

    # 5. Adicionar rótulos de dados (texto)
    # A lógica de anotação encontra isso
    text_labels = []
    # Só adiciona rótulos se a grade não for muito densa
    if rows * cols < 150 and random.random() < 0.8:
        val_min, val_max = data.min(), data.max()

        for r in range(rows):
            for c in range(cols):
                val = data[r, c]

                # Determina a cor do texto para contraste
                norm_val = (val - val_min) / (val_max - val_min) if val_max > val_min else 0
                text_color = 'white' if (norm_val < 0.4 and cmap_type=='sequential') or (0.2 < norm_val < 0.8 and cmap_type=='diverging') else 'black'

                # Formata o texto
                if np.issubdtype(data.dtype, np.integer):
                    text_str = f"{val:d}"
                else:
                    text_str = f"{val:.1f}"

                # Adiciona artista de texto
                txt = ax.text(c + 0.5, r + 0.5, text_str,
                              ha='center', va='center',
                              fontsize=8, color=text_color, zorder=10)
                text_labels.append(txt)

    other_artists.extend(text_labels)

    # 6. Configurar eixos (importante para heatmaps) - ENHANCED WITH DIVERSE LABELS
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_yticks(np.arange(rows) + 0.5)

    # ENHANCED: Select varied axis labels based on domain
    if is_scientific:
        xlabel = random.choice(HEATMAP_XLABELS_SCIENTIFIC)
        ylabel = random.choice(HEATMAP_YLABELS_SCIENTIFIC)
        colorbar_title = random.choice(COLORBAR_TITLES_SCIENTIFIC)
        chart_title = random.choice([t for t in HEATMAP_CHART_TITLES 
                                     if any(word in t.lower() for word in 
                                            ['gene', 'expression', 'sample', 'treatment', 
                                             'correlation', 'clustering', 'temporal'])])
        if not chart_title:  # Fallback if no matching titles found
            chart_title = random.choice(HEATMAP_CHART_TITLES)
    else:
        xlabel = random.choice(HEATMAP_XLABELS_BUSINESS)
        ylabel = random.choice(HEATMAP_YLABELS_BUSINESS)
        colorbar_title = random.choice(COLORBAR_TITLES_BUSINESS)
        chart_title = random.choice([t for t in HEATMAP_CHART_TITLES 
                                     if any(word in t.lower() for word in 
                                            ['performance', 'customer', 'sales', 'revenue',
                                             'market', 'product', 'regional', 'cohort'])])
        if not chart_title:  # Fallback if no matching titles found
            chart_title = random.choice(HEATMAP_CHART_TITLES)

    # Set labels with variety
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(chart_title)

    # ENHANCED: Generate varied tick labels based on xlabel/ylabel context
    if any(word in xlabel.lower() for word in ['time', 'day', 'week', 'quarter']):
        xticklabels = generate_temporal_labels(cols, xlabel)
    elif any(word in xlabel.lower() for word in ['dose', 'concentration', 'temperature']):
        xticklabels = generate_numeric_labels(cols, xlabel)
    elif any(word in xlabel.lower() for word in ['gene', 'protein', 'metabolite']):
        xticklabels = generate_biological_labels(cols, xlabel)
    elif any(word in xlabel.lower() for word in ['product', 'category', 'region']):
        xticklabels = generate_categorical_labels(cols, xlabel)
    else:
        xticklabels = [f"{xlabel.split()[0][:3]}{i+1}" for i in range(cols)]

    if any(word in ylabel.lower() for word in ['time', 'day', 'week', 'quarter']):
        yticklabels = generate_temporal_labels(rows, ylabel)
    elif any(word in ylabel.lower() for word in ['dose', 'concentration', 'temperature']):
        yticklabels = generate_numeric_labels(rows, ylabel)
    elif any(word in ylabel.lower() for word in ['gene', 'protein', 'metabolite']):
        yticklabels = generate_biological_labels(rows, ylabel)
    elif any(word in ylabel.lower() for word in ['product', 'category', 'region']):
        yticklabels = generate_categorical_labels(rows, ylabel)
    else:
        yticklabels = [f"{ylabel.split()[0][:3]}{i+1}" for i in range(rows)]

    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    # Girar rótulos do eixo x se houver muitos
    if cols > 10:
        ax.tick_params(axis='x', labelrotation=90)

    # ENHANCED: Add colorbar with varied title
    cbar.set_label(colorbar_title, rotation=270, labelpad=20)

    # Inverter eixo Y para corresponder ao layout da matriz (linha 0 no topo)
    ax.invert_yaxis()

    # Garantir que nenhuma linha de grade do tema interfira
    ax.grid(False)

    return data_artists, other_artists, [], 'vertical', [], [], \
           {'primary_scale_axis': 'none'}, None    # Inverter eixo Y para corresponder ao layout da matriz (linha 0 no topo)
    ax.invert_yaxis()
    
    # Garantir que nenhuma linha de grade do tema interfira
    ax.grid(False)

    return data_artists, other_artists, [], 'vertical', [], [], \
           {'primary_scale_axis': 'none'}, None

# Additional helper functions for completeness
def add_jitter_overlay(ax, bar_info, orientation='vertical'):
    """Add jitter points overlaid on bars for scientific visualization"""
    for info in bar_info:
        center_pos, mean_val, width_val = info['center'], info['height'], info['width']
        
        if mean_val == 0: 
            continue
        
        n_points = random.randint(8, 20)
        points = np.random.normal(loc=mean_val, scale=abs(mean_val) * random.uniform(0.1, 0.25), size=n_points)
        jitter = np.random.uniform(-width_val * 0.3, width_val * 0.3, size=n_points)
        
        if orientation == 'vertical':
            ax.scatter(center_pos + jitter, points, c='black', s=8, alpha=0.3, zorder=10)
        else:
            ax.scatter(points, center_pos + jitter, c='black', s=8, alpha=0.3, zorder=10)


def detect_inflection_points(x_data, y_data, threshold=0.1):
    """Detect inflection points using second derivative analysis."""
    if len(x_data) < 3:
        return []
    
    inflection_points = []
    y_range = max(y_data) - min(y_data)
    
    for i in range(1, len(y_data) - 1):
        d2y = y_data[i+1] - 2*y_data[i] + y_data[i-1]
        if abs(d2y) > threshold * y_range:
            inflection_points.append((x_data[i], y_data[i], i))
    
    return inflection_points

def detect_extrema(xdata, ydata, window_size=3, prominence_factor=0.05):
    """
    Detect local maxima (peaks) and minima (valleys) robustly.
    """
    if len(ydata) < 3:
        return [], []
    
    # Smooth with conservative sigma to preserve peak locations
    y_smooth = gaussian_filter(ydata, sigma=0.8)
    
    # Adaptive prominence based on data range
    y_range = np.max(ydata) - np.min(ydata)
    if y_range == 0:
        return [], []
    
    min_prominence = prominence_factor * y_range
    
    # Find peaks using scipy (robust to noise)
    peaks_idx, peak_props = find_peaks(
        y_smooth, 
        distance=max(1, window_size // 2),
        prominence=min_prominence
    )
    
    valleys_idx, valley_props = find_peaks(
        -y_smooth,
        distance=max(1, window_size // 2),
        prominence=min_prominence
    )
    
    # Sort by prominence and take top 2 of each
    if len(peaks_idx) > 2:
        peak_prominences = peak_props['prominences']
        top_peaks = peaks_idx[np.argsort(peak_prominences)[-2:]]
        peaks_idx = sorted(top_peaks)
    
    if len(valleys_idx) > 2:
        valley_prominences = valley_props['prominences']
        top_valleys = valleys_idx[np.argsort(valley_prominences)[-2:]]
        valleys_idx = sorted(top_valleys)
    
    # Return using ORIGINAL ydata for exact coordinates
    peaks = [(float(xdata[i]), float(ydata[i]), int(i)) for i in peaks_idx]
    valleys = [(float(xdata[i]), float(ydata[i]), int(i)) for i in valleys_idx]
    
    return peaks, valleys


def generate_perlin_heatmap(rows, cols, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate spatially coherent heatmap using Perlin noise.
    
    Args:
        rows, cols: Grid dimensions
        scale: Controls feature size (lower = larger features)
        octaves: Detail layers (higher = more detail)
        persistence: Amplitude decay per octave
        lacunarity: Frequency increase per octave
    """
    # This is a simplified version - we'll use scipy for actual Perlin noise
    # Generate base noise that creates spatially coherent patterns
    y, x = np.ogrid[:rows, :cols]
    
    # Create a base pattern using sum of sine waves with different frequencies
    data = np.zeros((rows, cols))
    
    # Add multiple frequency components to simulate Perlin noise characteristics
    for freq in [0.5, 1.0, 2.0]:
        x_scaled = x / (cols / freq)
        y_scaled = y / (rows / freq)
        # Combine sine and cosine waves to create complex patterns
        component = np.sin(x_scaled) * np.cos(y_scaled)
        data += component
    
    # Add some random variation
    random_noise = np.random.random((rows, cols)) * 0.2
    data = data + random_noise
    
    # Normalize to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())
    return data

def generate_correlated_blocks(rows, cols, num_blocks=3, block_corr=0.8, noise_level=0.1):
    """
    Generate heatmap with distinct correlated blocks.
    
    Args:
        num_blocks: Number of cluster groups
        block_corr: Within-block correlation (0-1)
        noise_level: Random noise proportion
    """
    # Assign rows/cols to blocks
    row_blocks = np.random.randint(0, num_blocks, rows)
    col_blocks = np.random.randint(0, num_blocks, cols)
    
    # Build correlation matrix with block structure
    corr_matrix = np.eye(rows)
    for block_id in range(num_blocks):
        block_rows = np.where(row_blocks == block_id)[0]
        
        # Set high correlation within blocks
        for i in block_rows:
            for j in block_rows:
                if i != j:
                    corr_matrix[i, j] = block_corr
    
    # Generate correlated data
    mean = np.zeros(rows)
    samples = np.random.multivariate_normal(mean=mean, cov=corr_matrix, size=cols).T
    
    # Ensure positive values and add noise
    data = np.abs(samples) + np.random.normal(0, noise_level, (rows, cols))
    
    # Normalize to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())
    return data

def generate_clustered_heatmap(rows, cols, num_clusters=5, cluster_spread=0.15, base_intensity=0.2):
    """
    Generate heatmap with distinct hot/cold clusters.
    Mimics genomic expression patterns or user activity zones.
    """
    # Random cluster centers
    cluster_centers = np.random.rand(num_clusters, 2)
    cluster_centers[:, 0] *= rows
    cluster_centers[:, 1] *= cols
    
    # Random cluster intensities
    cluster_intensities = np.random.uniform(0.3, 1.0, num_clusters)
    
    # Create coordinate grid
    coords = np.array([[i, j] for i in range(rows) for j in range(cols)])
    
    # Calculate distances to all clusters
    distances = cdist(coords, cluster_centers)
    
    # Compute influence of each cluster (Gaussian decay)
    influences = np.exp(-distances**2 / (2 * (cluster_spread * min(rows, cols))**2))
    weighted_influences = influences * cluster_intensities
    
    # Sum influences
    data = weighted_influences.sum(axis=1).reshape(rows, cols)
    
    # Add base intensity and noise
    data = base_intensity + data * (1 - base_intensity)
    data += np.random.normal(0, 0.05, (rows, cols))
    
    return np.clip(data, 0, 1)

def generate_structured_heatmap(heatmap_type='auto', size=None, debug_mode=False):
    """
    Gera heatmaps com estruturas realistas, conforme a análise do usuário.
    Tipos: 'correlation', 'clustered', 'timeseries', 'confusion'.
    """
    
    if heatmap_type == 'auto':
        heatmap_type = np.random.choice(['correlation', 'clustered', 'timeseries', 'confusion', 'perlin', 'correlated_blocks', 'clustered_patterns'],
                                      p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10])
    
    if debug_mode:
        print(f"DEBUG [HEATMAP_GEN] Gerando tipo: {heatmap_type}")

    if heatmap_type == 'correlation':
        # Matriz de correlação (simétrica, -1 a 1)
        n = size or np.random.randint(8, 20)
        A = np.random.randn(n, n)
        cov = A @ A.T  # Positivo semi-definido
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)  # Autocorrelação perfeita
        return corr, 'diverging'  # Usar colormap RdBu ou coolwarm

    elif heatmap_type == 'clustered':
        # Estrutura de blocos (expressão gênica, segmentos de clientes)
        rows, cols = size or (np.random.randint(15, 30), np.random.randint(10, 20))
        num_clusters = np.random.randint(2, 5)
        data = np.zeros((rows, cols))
        
        cluster_size = rows // num_clusters
        for i in range(num_clusters):
            start = i * cluster_size
            end = min((i + 1) * cluster_size, rows)
            base_value = np.random.uniform(20, 80)
            noise_level = np.random.uniform(5, 15)
            data[start:end, :] = np.random.normal(base_value, noise_level, (end - start, cols))
        
        # Adiciona transições suaves entre clusters
        data = gaussian_filter(data, sigma=0.5)
        return data, 'sequential'  # Usar viridis ou plasma

    elif heatmap_type == 'timeseries':
        # Heatmap de série temporal (transições temporais suaves)
        rows, cols = size or (np.random.randint(10, 20), np.random.randint(20, 40))
        data = np.zeros((rows, cols))
        
        for i in range(rows):
            # Cada linha é uma série temporal com tendência + ruído
            trend = np.linspace(np.random.uniform(10, 40), 
                                np.random.uniform(50, 90), cols)
            seasonal = 10 * np.sin(np.linspace(0, 2*np.pi*np.random.randint(1, 4), cols))
            noise = np.random.normal(0, 5, cols)
            data[i, :] = trend + seasonal + noise
            
        return data, 'sequential'

    elif heatmap_type == 'confusion':
        # Matriz de confusão (concentrada na diagonal)
        n = size or np.random.randint(5, 15)
        data = np.random.uniform(0, 10, (n, n))
        # Aumenta a diagonal (classificações corretas)
        np.fill_diagonal(data, np.random.uniform(50, 100, n))
        # Adiciona alguma confusão entre classes adjacentes
        for i in range(n-1):
            data[i, i+1] += np.random.uniform(10, 30)
            data[i+1, i] += np.random.uniform(10, 30)
        # Retorna como inteiros
        return data.astype(int), 'sequential'
    
    # Fallback (caso o tipo seja desconhecido)
    data = np.random.rand(10, 10) * 100
    return data, 'sequential'


def generate_temporal_labels(n, label_type):
    """Generate varied temporal labels"""
    if 'quarter' in label_type.lower():
        return [f"Q{(i % 4) + 1}'{20 + i//4}" for i in range(n)]
    elif 'week' in label_type.lower():
        return [f"W{i+1}" for i in range(n)]
    elif 'day' in label_type.lower():
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return [days[i % 7] for i in range(n)]
    elif 'hour' in label_type.lower() or '(h)' in label_type:
        return [f"{i}h" if i < 24 else f"{i}h" for i in range(n)]
    else:
        return [f"T{i+1}" for i in range(n)]

def generate_numeric_labels(n, label_type):
    """Generate numeric labels with units"""
    if 'dose' in label_type.lower() or 'concentration' in label_type.lower():
        base = np.logspace(-1, 2, n)
        return [f"{v:.1f}" for v in base]
    elif 'temperature' in label_type.lower():
        temps = np.linspace(20, 40, n)
        return [f"{t:.0f}°" for t in temps]
    else:
        return [f"{i*10}" for i in range(n)]

def generate_biological_labels(n, label_type):
    """Generate biological entity labels"""
    prefixes = ['BRCA', 'TP53', 'EGFR', 'MYC', 'KRAS', 'ALK', 'RET', 'MET']
    if 'gene' in label_type.lower():
        return [f"{random.choice(prefixes)}{random.randint(1,5)}" for _ in range(n)]
    elif 'protein' in label_type.lower():
        return [f"P{random.randint(10000,99999)}" for _ in range(n)]
    else:
        return [f"Bio{i+1}" for i in range(n)]

def generate_categorical_labels(n, label_type):
    """Generate business/categorical labels"""
    if 'product' in label_type.lower():
        return [f"SKU-{random.randint(1000,9999)}" for _ in range(n)]
    elif 'region' in label_type.lower():
        regions = ['North', 'South', 'East', 'West', 'Central']
        return [f"{random.choice(regions)}-{i+1}" for i in range(n)]
    else:
        return [f"Cat{i+1}" for i in range(n)]

def generate_temporal_labels(n, label_type):
    """Generate varied temporal labels"""
    if 'quarter' in label_type.lower():
        return [f"Q{(i % 4) + 1}'{20 + i//4}" for i in range(n)]
    elif 'week' in label_type.lower():
        return [f"W{i+1}" for i in range(n)]
    elif 'day' in label_type.lower():
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return [days[i % 7] for i in range(n)]
    elif 'hour' in label_type.lower() or '(h)' in label_type:
        return [f"{i}h" if i < 24 else f"{i}h" for i in range(n)]
    else:
        return [f"T{i+1}" for i in range(n)]

def generate_numeric_labels(n, label_type):
    """Generate numeric labels with units"""
    if 'dose' in label_type.lower() or 'concentration' in label_type.lower():
        base = np.logspace(-1, 2, n)
        return [f"{v:.1f}" for v in base]
    elif 'temperature' in label_type.lower():
        temps = np.linspace(20, 40, n)
        return [f"{t:.0f}°" for t in temps]
    else:
        return [f"{i*10}" for i in range(n)]

def generate_biological_labels(n, label_type):
    """Generate biological entity labels"""
    prefixes = ['BRCA', 'TP53', 'EGFR', 'MYC', 'KRAS', 'ALK', 'RET', 'MET']
    if 'gene' in label_type.lower():
        return [f"{random.choice(prefixes)}{random.randint(1,5)}" for _ in range(n)]
    elif 'protein' in label_type.lower():
        return [f"P{random.randint(10000,99999)}" for _ in range(n)]
    else:
        return [f"Bio{i+1}" for i in range(n)]

def generate_categorical_labels(n, label_type):
    """Generate business/categorical labels"""
    if 'product' in label_type.lower():
        return [f"SKU-{random.randint(1000,9999)}" for _ in range(n)]
    elif 'region' in label_type.lower():
        regions = ['North', 'South', 'East', 'West', 'Central']
        return [f"{random.choice(regions)}-{i+1}" for i in range(n)]
    else:
        return [f"Cat{i+1}" for i in range(n)]

def calculate_pie_geometry(wedges, ax, debug_mode=False):
    """
    Calculate geometric keypoints for pie chart.
    CRITICAL FIX: Use wedge.center (displaced center) for arc calculations.
    CRITICAL FIX: True original center is (0,0) in data coordinates.
    """
    if not wedges:
        return None
    
    if debug_mode:
        print(f"DEBUG [PIE_GEOM] Calculating geometry for {len(wedges)} wedges")
    
    # CRITICAL FIX: The true, non-exploded center of a matplotlib pie chart
    # drawn at the origin is (0.0, 0.0) in data coordinates.
    original_centerx = 0.0
    original_centery = 0.0
    
    wedge_geometry = []
    
    for idx, wedge in enumerate(wedges):
        # CRITICAL: Use this wedge's specific center.
        # This is (0,0) for non-exploded wedges and (dx, dy) for exploded ones.
        wedge_cx, wedge_cy = wedge.center 
        
        radius = wedge.r
        theta1 = np.deg2rad(wedge.theta1)
        theta2 = np.deg2rad(wedge.theta2)
        thetamid = (theta1 + theta2) / 2
        
        # --- INÍCIO DA MODIFICAÇÃO ---
        # Calcular pontos intermediários a 1/3 e 2/3 do arco
        theta_inter_1 = theta1 + (theta2 - theta1) / 3.0
        theta_inter_2 = theta1 + 2 * (theta2 - theta1) / 3.0
        # --- FIM DA MODIFICAÇÃO ---
        
        # CRITICAL FIX: Calculate arc_boundary with proper sampling
        angle_span = wedge.theta2 - wedge.theta1
        num_arc_points = max(5, int(angle_span / 15))  # 1 point per 15 degrees
        theta_samples = np.linspace(theta1, theta2, num_arc_points)
        
        # CRITICAL FIX: Use the wedge's specific center (wedge_cx, wedge_cy)
        # for all arc point calculations.
        arc_boundary_points = [
            (float(wedge_cx + radius * np.cos(theta)), 
             float(wedge_cy + radius * np.sin(theta)))
            for theta in theta_samples
        ]
        
        wedge_geometry.append({
            'wedge_idx': idx,
            'center': (float(wedge_cx), float(wedge_cy)), # Displaced center
            'original_center': (original_centerx, original_centery), # TRUE center (0,0)
            'radius': float(radius),
            'theta1_deg': float(wedge.theta1),
            'theta2_deg': float(wedge.theta2),
            'arc_start': (
                float(wedge_cx + radius * np.cos(theta1)), # Use displaced center
                float(wedge_cy + radius * np.sin(theta1))
            ),
            'arc_end': (
                float(wedge_cx + radius * np.cos(theta2)), # Use displaced center
                float(wedge_cy + radius * np.sin(theta2))
            ),
            # --- INÍCIO DA MODIFICAÇÃO ---
            'arc_inter_1': (
                float(wedge_cx + radius * np.cos(theta_inter_1)), # Use displaced center
                float(wedge_cy + radius * np.sin(theta_inter_1))
            ),
            'arc_inter_2': (
                float(wedge_cx + radius * np.cos(theta_inter_2)), # Use displaced center
                float(wedge_cy + radius * np.sin(theta_inter_2))
            ),
            # --- FIM DA MODIFICAÇÃO ---
            'arc_mid': (
                float(wedge_cx + radius * np.cos(thetamid)), # Use displaced center
                float(wedge_cy + radius * np.sin(thetamid))
            ),
            'arc_boundary': arc_boundary_points,  # FIXED: Now correctly calculated
            'wedge_label_point': (
                float(wedge_cx + radius * 0.7 * np.cos(thetamid)),
                float(wedge_cy + radius * 0.7 * np.sin(thetamid))
            ),
            'angle_span': float(angle_span)
        })
        
        if debug_mode:
            print(f"DEBUG [PIE_GEOM] Wedge {idx}: Center=({wedge_cx:.2f},{wedge_cy:.2f}), R={radius:.2f}")
            print(f"DEBUG [PIE_GEOM] Wedge {idx}: ArcStart=({wedge_geometry[-1]['arc_start'][0]:.2f},{wedge_geometry[-1]['arc_start'][1]:.2f})")
    
    return {
        'center_point': (original_centerx, original_centery), # Return TRUE center
        'wedges': wedge_geometry
    }


def extract_scale_axis_info(ax, chart_type_str):
    """
    Extract information about scale axes (primary and secondary) for the chart.
    """
    # Determine primary and secondary scale axes based on chart type and orientation
    if chart_type_str in ['bar', 'histogram']:
        # For bar charts, primary scale axis is typically the value axis (y-axis for vertical, x-axis for horizontal)
        if hasattr(ax, '_orientation') and ax._orientation == 'horizontal':
            primary_scale_axis = 'x'
            secondary_scale_axis = 'y' if ax.yaxis_inverted() else None
        else:
            primary_scale_axis = 'y'  # Default to y-axis for vertical bars
            secondary_scale_axis = 'x' if ax.xaxis_inverted() else None
    elif chart_type_str in ['line', 'area', 'scatter']:
        # For line charts, primary scale axis is typically y-axis
        primary_scale_axis = 'y'
        secondary_scale_axis = 'x'
    else:
        # Default for other chart types
        primary_scale_axis = 'y'
        secondary_scale_axis = None

    return {
        "primary_scale_axis": primary_scale_axis,
        "secondary_scale_axis": secondary_scale_axis
    }


def extract_bar_info(ax, chart_type_str):
    """
    Extract detailed information about bars in bar charts and histograms.
    """
    if chart_type_str not in ['bar', 'histogram']:
        return []

    bar_info_list = []

    # Iterate through all patches in the axes to find bar rectangles
    for i, patch in enumerate(ax.patches):
        if hasattr(patch, 'get_xy') and hasattr(patch, 'get_width') and hasattr(patch, 'get_height'):
            x, y = patch.get_xy()
            width = patch.get_width()
            height = patch.get_height()

            # Calculate center based on orientation
            center_x = x + width / 2
            center_y = y + height / 2

            bar_info = {
                "center": float(center_x) if width > height else float(center_y),
                "height": float(height),
                "width": float(width),
                "bottom": float(y),
                "top": float(y + height),
                "series_idx": 0,  # Would need to determine series in multi-series charts
                "bar_idx": i,
                "axis": "primary"  # Default to primary axis
            }

            # Determine if it's horizontal or vertical based on dimensions
            if width > height:
                bar_info["orientation"] = "horizontal"
            else:
                bar_info["orientation"] = "vertical"

            bar_info_list.append(bar_info)

    return bar_info_list


def extract_keypoint_info(ax, chart_type_str):
    """
    Extract keypoint information for line, area, and pie charts.
    """
    if chart_type_str not in ['line', 'area', 'pie']:
        return []

    keypoint_info_list = []

    # For line and area charts, extract line data points
    for i, line in enumerate(ax.lines):
        if hasattr(line, 'get_data'):
            x_data, y_data = line.get_data()

            # Get inflection points if possible
            inflection_indices = []
            if len(y_data) > 2:
                # Simple inflection detection (where second derivative changes sign)
                y_diff = np.diff(y_data)
                y_diff2 = np.diff(y_diff)
                inflection_indices = np.where(y_diff2[:-1] * y_diff2[1:] < 0)[0] + 1  # +1 because diff reduces length by 1

            points = []
            for j, (x, y) in enumerate(zip(x_data, y_data)):
                is_inflection = j in inflection_indices
                points.append({
                    "x": float(x),
                    "y": float(y),
                    "is_inflection": bool(is_inflection)
                })

            keypoint_info_list.append({
                "series_idx": i,
                "points": points
            })

    return keypoint_info_list


def extract_boxplot_metadata(ax, chart_type_str):
    """
    Extract metadata for boxplot charts.
    """
    if chart_type_str != 'box':
        return {}

    # Look for boxplot elements in the axes
    box_metadata = {
        "num_groups": 0,
        "box_width": 0.0,
        "orientation": "vertical",  # Default
        "medians": []
    }

    # Find boxplot elements by looking for specific artists
    median_lines = []
    for line in ax.lines:
        # Matplotlib boxplots typically have specific line styles for median lines
        if hasattr(line, 'get_color'):
            # Check if this might be a median line by its properties
            x_data, y_data = line.get_data()
            if len(x_data) == 2 and len(y_data) == 2:
                # A median line is typically a horizontal line segment
                if abs(y_data[0] - y_data[1]) < 0.01:  # Almost same y values
                    median_lines.append(line)

    medians = []
    for i, line in enumerate(median_lines):
        x_data, y_data = line.get_data()
        median_x = np.mean(x_data)
        median_y = np.mean(y_data)

        medians.append({
            "group_index": i,
            "group_label": f"Group_{i}",
            "median_value": float(median_y),
            "lower_left": {"x": float(min(x_data)), "y": float(min(y_data))},
            "upper_right": {"x": float(max(x_data)), "y": float(max(y_data))},
            "center_x": float(median_x),
            "center_y": float(median_y),
            "line_length": float(abs(x_data[1] - x_data[0]))
        })

    box_metadata["num_groups"] = len(medians)
    box_metadata["medians"] = medians

    return box_metadata


def extract_pie_geometry(ax, chart_type_str):
    """
    Extract geometric information for pie charts.
    """
    if chart_type_str != 'pie':
        return {}

    # Look for wedge patches which represent pie slices
    wedges = []
    for i, patch in enumerate(ax.patches):
        if hasattr(patch, 'theta1') and hasattr(patch, 'theta2'):
            # This is likely a wedge/pie slice
            center_x, center_y = patch.center
            radius = patch.r
            start_angle = patch.theta1
            end_angle = patch.theta2
            mid_angle = (start_angle + end_angle) / 2

            wedges.append({
                "wedge_index": i,
                "start_angle": float(start_angle),
                "end_angle": float(end_angle),
                "mid_angle": float(mid_angle),
                "percentage": float((end_angle - start_angle) / 360.0 * 100)
            })

    # For pie charts, the center is typically (0, 0) unless offset
    center_point = {"x": 0.0, "y": 0.0}
    if ax.patches:
        # Use the center of the first patch as the pie center
        first_patch = ax.patches[0]
        if hasattr(first_patch, 'center'):
            center_x, center_y = first_patch.center
            center_point = {"x": float(center_x), "y": float(center_y)}

    return {
        "center_point": center_point,
        "radius": float(radius if 'radius' in locals() else 0.5),
        "wedges": wedges
    }


