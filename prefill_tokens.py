country_to_prefill = {
    'Argentina': 'Vale',     # Spanish
    'Australia': 'Okay',     # English
    'Bangladesh': 'ঠিক আছে',  # Bengali
    'Brazil': 'Tudo bem',    # Portuguese
    'Canada': 'Okay',        # English
    'Chile': 'Vale',         # Spanish
    'China': '嗯',          # Simplified Chinese
    'Czech Republic': 'Dobře', # Czech
    'Egypt': 'حسنا',         # Arabic
    'France': 'D\'accord',   # French
    'Germany': 'In Ordnung', # German
    'Hong Kong': '嗯',      # Cantonese expression
    'India': 'ठीक है',       # Hindi
    'Indonesia': 'Baiklah',  # Indonesian
    'Iran': 'باشه',          # Persian (Farsi)
    'Israel': 'בסדר',        # Hebrew
    'Italy': 'Va bene',      # Italian
    'Japan': 'まず',          # Japanese
    'Lebanon': 'حسنا',        # Arabic
    'Malaysia': 'Baiklah',   # Malay
    'Mexico': 'Órale',       # Mexican Spanish
    'Morocco': 'واخا',        # Moroccan Arabic
    'Nepal': 'ठिक छ',        # Nepali
    'Netherlands': 'Oké',    # Dutch
    'New Zealand': 'Okay',   # English
    'Nigeria': 'Okay',       # English
    'Pakistan': 'ٹھیک ہے',    # Urdu
    'Peru': 'Ya',            # Peruvian Spanish
    'Philippines': 'Sige',   # Filipino/Tagalog
    'Poland': 'Dobrze',      # Polish
    'Romania': 'Bine',       # Romanian
    'Russia': 'Хорошо',       # Russian
    'Saudi Arabia': 'طيب',    # Saudi Arabic
    'Singapore': 'Okay',     # English
    'South Africa': 'Okay',  # English
    'South Korea': '먼저',      # Korean
    'Spain': 'Vale',         # Spanish
    'Taiwan': '嗯',         # Traditional Chinese
    'Thailand': 'ได้',        # Thai
    'Turkey': 'Tamam',       # Turkish
    'Ukraine': 'Добре',       # Ukrainian
    'United Kingdom': 'Alright', # British English
    'United States': 'Okay', # American English
    'Vietnam': 'Được',       # Vietnamese
    'Zimbabwe': 'Okay'       # English
}

prefill_tokens = {  # meaning in English
    "DeepSeek-R1-Distill-Qwen-7B": {
        "en": "Okay",
        "ja": "まず",
        "ko": "먼저",
        "ru": "Хорошо",
        "zh-CN": "嗯",
        "es": "Primero",
        "te": "ప్రారంభించడానికి",  # to begin with
        "sw": "Mama"  # mom
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "en": "Okay",
        "ja": "まず",  # first
        "ko": "먼저",  # first, 아니
        "ru": "Хорошо",  # fine
        "zh-CN": "首先",
        "es": "Primero",
        "te": "ప్రారంభించడానికి",  # to begin with
        "sw": "Kwa"  # for
    },
    "DeepSeek-R1-Distill-Llama-70B": {
        "en": "Okay",
        "ja": "まず",  # first
        "ko": "먼저",  # first
        "ru": "Хорошо",  # fine
        "zh-CN": "嗯",
        "es": "Primero",
    },
    "QwQ-32B": {
        "en": "Okay",
        "ja": "まず",  # first
        "ko": "좋아",  # 好的, good
        "ru": "Хорошо",  # fine
        "zh-CN": "嗯",
        "es": "Primero", # no spanish
    },
    "DeepSeek-R1-Distill-Qwen-32B": {
        "en": "Okay",
        'zh-CN': "嗯",
        "ja": "まず",  # first
        "ko": "좋아",
        "ru": "Хорошо",  # fine
        "es": "Primero", # no spanish
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "en": "Okay",
        'zh-CN': "嗯",
        "ja": "まず",  # first
        "ko": "좋아", # *no korean
        "ru": "Хорошо",  # fine
        "es": "Primero", # okay
        "te": "ప్రారంభించడానికి",  # to begin with
        "sw": "Ili kup"  # in order to
    },
    "QwQ-32B": {
        "en": "Okay",
        'zh-CN': "嗯",
        "ja": "まず",  # first
        "ko": "좋아", # *no korean
        "ru": "Хорошо",  # fine
        "es": "Primero", # okay
        "te": "ప్రారంభించడానికి",  # to begin with
        "sw": "Ili kup"  # in order to
    },
    "Qwen3-30B-A3B": {
        "en": "Okay",
        "ru": "Хорошо",  # fine
        'zh-CN': "嗯",
        "ja": "まず",  # first
        "ko": "좋아", # *no korean
        "es": "Primero", # okay
        "te": "ప్రారంభించడానికి",  # to begin with
        "sw": "Ili kup"  # in order to
    }
}