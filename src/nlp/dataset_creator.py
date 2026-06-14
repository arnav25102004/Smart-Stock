"""
Multilingual NLP dataset generator for SMARTSTOCK.

Generates a CSV with 1,000+ realistic inventory-management sentences in
English, Hindi (Romanized), Kannada (Romanized), and Code-Mixed variants
across 5 intents: check_stock, add_stock, get_prediction, low_stock_alert, help.

Sentences simulate how Indian shopkeepers actually type on phone keyboards —
Romanized scripts, informal style, abbreviations, and code-mixing.

Usage:
    python src/nlp/dataset_creator.py
"""

import os
import random
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

NLP_DIR = "data/nlp_dataset"
os.makedirs(NLP_DIR, exist_ok=True)

# ─────────────────────────── Sentence Templates ───────────────────────────

SENTENCES = {
    # ──────────────── ENGLISH ────────────────
    "English": {
        "check_stock": [
            "how much rice is left", "check stock of sugar", "what is the current dal stock",
            "is milk available", "show me soap inventory", "how much oil remaining",
            "flour stock level", "tea stock please", "what items do we have",
            "check inventory", "current stock of rice", "how many bags of sugar",
            "do we have enough milk", "remaining wheat quantity", "salt stock check",
            "biscuit inventory now", "how much butter is there", "check ghee level",
            "show current stock", "what is left in store", "remaining cooking oil",
            "check maida stock", "how much coffee is there", "stock update for all items",
            "do we have detergent", "shampoo stock level", "check soap quantity",
            "current rice inventory", "tell me stock of all products",
            "how many units of dal", "what is our current flour stock",
            "how much cheese do we have", "check the egg stock",
            "is curd available", "how much paneer is left",
            "check bread quantity", "biscuits remaining count",
            "how many packets of noodles", "check vegetable oil stock",
            "sugar quantity check", "remaining stock report",
            "inventory summary please", "stock levels now", "check all items",
            "is there enough atta", "onion price and stock", "potato stock check",
            "current tomato count", "how much besan is there", "check namkeen stock",
            "murmura quantity", "how many cold drinks left", "juice packets remaining",
            "check masala stock", "spices inventory level", "check chili powder",
            "remaining turmeric stock", "coriander stock level", "check pepper stock",
            "how much cardamom left", "cloves remaining count", "check all spice levels",
            "rice bag count", "dal packet count", "wheat flour remaining",
            "stock check for today", "morning inventory check", "evening stock count",
            "weekly stock check", "check low items first", "show me all stock levels",
            "inventory for tomorrow", "quick stock check", "what needs restocking",
            "which items are running", "most used items stock", "fast moving items check",
            "slow moving items stock", "check expired items", "near expiry items check",
            "current stock value", "total inventory count", "check cold storage items",
        ],
        "add_stock": [
            "add 10 kg rice", "restock sugar please", "we need more milk",
            "order more dal", "refill cooking oil", "put 5 kg flour",
            "add soap to inventory", "increase ghee stock", "reorder tea",
            "we are running out of sugar so order more", "add new stock of butter",
            "purchase 20 kg wheat", "bring more biscuits", "add coffee to store",
            "restock all vegetables", "place order for rice", "add 50 packets of noodles",
            "update stock with new delivery", "add received goods to inventory",
            "new stock arrived please add", "replenish masala stock",
            "add 100 units of soap", "increase salt stock by 10 kg",
            "add bread to inventory", "we need to reorder eggs",
            "add 5 litre oil cans", "increase juice stock", "add cold drink bottles",
            "reorder spices", "add delivered items now", "put new stock in",
            "add stock for upcoming festival", "increase stock before weekend",
            "order 25 kg atta", "add washing powder to store",
            "add 10 kg besan", "restocking needed immediately",
            "add new batch of goods", "update inventory after delivery",
            "add received items to system", "increase all low items",
        ],
        "get_prediction": [
            "how much rice will sell tomorrow", "predict demand for sugar next week",
            "forecast dal sales", "what will be milk demand", "predict oil consumption",
            "expected sales for flour tomorrow", "demand forecast please",
            "how much should I order for next week", "predict festive season demand",
            "forecast for this weekend", "what will sell most tomorrow",
            "predict sales for monday", "demand estimate for sugar",
            "forecast next month inventory need", "predict high demand items",
            "what stock should I keep for next 7 days", "sales prediction for rice",
            "expected demand this week", "predict top selling items",
            "forecast for upcoming holiday", "predict tomorrow's need",
            "how much oil will I need this week", "demand forecast for all items",
            "predict slow moving items", "what will be needed on sunday",
        ],
        "low_stock_alert": [
            "which items are low", "alert me for low stock", "low stock warning",
            "what is running out", "items below minimum level", "critical stock alert",
            "nearly empty items", "what needs urgent reorder", "stock shortage warning",
            "tell me items that are almost finished", "running low on what items",
            "minimum stock threshold crossed", "send low stock notification",
            "check for empty shelves", "urgent stock alert needed",
            "which products are critically low", "zero stock items",
            "items that will finish today", "stock shortage list please",
            "out of stock alert", "emergency reorder needed for which items",
            "items running out fast", "daily low stock report", "what is almost over",
            "shortage items today",
        ],
        "help": [
            "how do I use this", "help me with inventory", "show guide",
            "how to check stock", "tutorial please", "what can you do",
            "how to add items", "guide for adding stock", "help with forecasting",
            "how does prediction work", "usage instructions", "show me how to start",
            "what are the features", "help menu", "what commands are available",
            "how to set minimum stock", "how to view reports", "help with alerts",
            "how to update inventory", "getting started guide", "how to use the app",
            "explain stock prediction", "help with this system", "what is smartstock",
            "how to get started",
        ],
    },

    # ──────────────── HINDI (Romanized) ────────────────
    "Hindi": {
        "check_stock": [
            "chawal kitna bacha hai", "rice ka stock kya hai", "kitna maal pada hai",
            "cheeni kitni bachi", "dal ka stock check karo", "doodh hai kya aaj",
            "sabun kitna hai", "tel kitna bacha", "aata ka stock batao",
            "chai patti kitni hai", "chawal kitna hai abhi", "sugar ka maal kitna",
            "ghee kitna bacha hai", "namak kitna pada hai", "biscuit ka stock dikhao",
            "maida kitni bachi hai", "coffee kitna hai", "makhan kitna hai",
            "sabhi cheez ka stock batao", "kya stock bacha hai",
            "dal packat kitne hain", "atta ka stock check karo",
            "puri inventory dikhao", "kya kya bacha hai store mein",
            "kitne bag chawal ke hain", "sabun ke packet kitne hain",
            "tel ka dabba kitna bacha", "chai ka packet kitna",
            "cheeni ka bora kitna", "ration kitna bacha hai",
            "stock update karo", "abhi kya available hai",
            "masala ka stock check karo", "mirch kitna hai",
            "haldi kitni bachi", "dhaniya kitna bacha hai",
            "saunf kitna hai abhi", "jeera kitna bacha",
            "stock ki list dikhao", "sab cheez ka stock dikhao",
            "maal kitna hai dukaan mein", "store mein kya bacha hai",
            "aaj ka stock check karo", "subah stock count karo",
            "poora inventory batao", "kya khatam hone wala hai",
            "fast chal raha kya item", "kaunsa maal jyada bikta hai",
            "stock status batao", "kya sab theek hai",
            "check karo dukaan ka maal", "dal ka kitna stock bacha",
            "besan kitna hai", "suji kitni bachi hai",
            "rava kitna pada hai", "poha kitna hai",
            "sago kitna bacha", "sabudana kitna hai",
            "peanut kitna bacha hai", "kaju kitne hain",
            "badam ka stock kitna", "kishmish kitni bachi",
            "namkeen kitni bachi hai", "chips ka stock kitna",
            "juice kitna hai", "cold drink kitni bachi",
            "pani ki bottle kitni", "shampoo kitna bacha hai",
            "sabun kitna bacha dukaan mein", "washing powder kitna",
            "detergent kitna bacha hai", "phenyl kitna hai",
            "agarbatti kitni bachi", "matchbox kitne hain",
        ],
        "add_stock": [
            "10 kg chawal add karo", "cheeni ka stock badhao",
            "doodh order karo", "dal mangao", "tel ka dabba aaya hai daldo",
            "aata 5 kg daalo", "sabun order karo", "ghee badhao stock mein",
            "chai mangao", "sugar khatam ho raha hai order karo",
            "makhan ka naya stock daalo", "20 kg gehu kharido",
            "biscuit mangao", "coffee ka stock badhao",
            "sabzi ka stock badhao", "chawal ka order do",
            "50 noodle packet dalo", "naya maal aaya hai add karo",
            "delivery aayi hai update karo", "naya stock daalo system mein",
            "masala bharo", "100 sabun dalo", "namak 10 kg badhao",
            "bread ka stock daalo", "ande mangao",
            "5 litre tel ka tin daalo", "juice badhao",
            "cold drink order karo", "masale order karo",
            "aaya maal system mein dalo", "naya maal daalo",
            "festival ke liye stock badhao", "weekend ke pehle stock badhao",
            "25 kg atta order karo", "kapde dhone ka powder daalo",
            "10 kg besan daalo", "jaldi restock karo",
            "naya batch daalo", "delivery ke baad update karo",
            "mala mili hai add karo",
        ],
        "get_prediction": [
            "kal kitna chawal bikega", "agle hafte sugar ki demand kya hogi",
            "dal ka forecast batao", "doodh ki demand kya hogi",
            "tel ki khapat batao", "kal ke liye aata kitna chahiye",
            "demand forecast chahiye", "agle hafte kitna order karu",
            "tyohar mein kitna bikega", "weekend mein kya jyada bikega",
            "kal kya best seller hoga", "somwar ke liye forecast batao",
            "sugar ki mang ka andaza lagao", "agle mahine ke liye stock plan",
            "kaun sa item jyada bikta hai batao", "7 din ke liye stock kitna rakhun",
            "chawal ki bikri ka andaza do", "is hafte ki umeed kya hai",
            "sabse jyada bikne wale item batao", "chutti ke din ke liye forecast",
            "kal ka hisab lagao", "is hafte tel kitna lagega",
            "sab items ki demand batao", "slow item batao",
            "itwaar ko kya chahiye hoga",
        ],
        "low_stock_alert": [
            "kaunsa item kam ho raha hai", "low stock ki suchi do",
            "kya khatam ho raha hai", "minimum se neeche kya hai",
            "stock khatam hone wala kaun sa", "jaldi order chahiye kaun se item mein",
            "kya khatam hone wala hai alert do", "khali shelf kaun si",
            "urgent order kaun sa item", "aaj kya finish hoga",
            "stock warning do", "kaunsa maal toot gaya",
            "kaun sa item bohot kam bacha hai", "aaj reorder kya karna",
            "zero stock kaun sa hai", "emergency mein kya order karu",
            "jaldi khatam ho raha kaunsa", "roz ka low stock report",
            "kya almost khatam hai", "kaun sa maal nahi bacha",
            "aaj kitna khatam hua", "stock alert do",
            "aaj ka shortage batao", "kya urgent hai",
            "sabse kam stock kaunsa",
        ],
        "help": [
            "kaise use karu", "mujhe madad chahiye", "guide dikhao",
            "stock kaise check karein", "tutorial chahiye", "kya kar sakte ho",
            "item kaise daale", "stock add karne ka tarika batao",
            "forecast kaise kaam karta hai", "prediction samjhao",
            "use karne ka tarika batao", "shuruat kaise kare",
            "kya features hain", "help menu dikhao", "kya kya kar sakte hai",
            "minimum stock kaise set kare", "report kaise dekhe",
            "alert kaise set kare", "inventory update kaise kare",
            "naya hu kaise shuru karu", "app kaise chalayein",
            "stock prediction kya hoti hai", "help chahiye",
            "smartstock kya hai", "kaise start karu",
        ],
    },

    # ──────────────── KANNADA (Romanized) ────────────────
    "Kannada": {
        "check_stock": [
            "akki eshtu ide", "sakkare eshtu uliyitu", "bele stock eshtu",
            "haalu ide ya", "sabbu eshtu ide", "enne eshtu uliyitu",
            "hittu stock heli", "tea powder eshtu ide", "dal eshtu baaki ide",
            "stock check maadi", "akki cheelagalu eshtu", "sakkare cheelagalu eshtu",
            "tuppada stock eshtu ide", "uppu eshtu ide", "biscuit stock tilikoli",
            "maida eshtu uliyitu", "coffee eshtu ide", "benne eshtu baaki",
            "ella item stock heli", "yenu stock ide",
            "dal packet eshtu ide", "hittu stock tilikoli",
            "full inventory tilikoli", "dudde yaavudu ide",
            "akki chaalu eshtu", "sabbu packet eshtu ide",
            "enne tin eshtu uliyitu", "chai packet eshtu",
            "sakkare cheelagalu eshtu", "rashana eshtu uliyitu",
            "stock update maadi", "enu available ide",
            "masala stock check maadi", "menasinakai eshtu ide",
            "arisina eshtu uliyitu", "kothambari eshtu baaki",
            "sompu eshtu ide", "jeerige eshtu uliyitu",
            "stock list tilikoli", "ella item stock tilikoli",
            "angadi saamanu eshtu", "store alli enu uliyitu",
            "indina stock check", "beleggina stock count",
            "poorna inventory heli", "enu mugiyutte ide",
            "jasti bikuna yaavudu", "yaav item jasti biku",
            "stock sthithi heli", "ella sari ide ya",
            "angadi saamanu check maadi", "bele eshtu uliyitu",
            "kadala bele eshtu ide", "rave eshtu uliyitu",
            "avalakki eshtu ide", "sabudana eshtu baaki",
            "kadale eshtu ide", "godambi eshtu ide",
            "badam stock eshtu", "kismis eshtu uliyitu",
            "namkeen eshtu uliyitu", "chips stock eshtu",
            "juice eshtu ide", "cool drink eshtu uliyitu",
            "neerinda bottle eshtu", "shampoo eshtu uliyitu",
            "sabbu eshtu uliyitu angadiyalli", "washing powder eshtu",
            "detergent eshtu ide", "phenyl eshtu uliyitu",
            "agarbatti eshtu ide", "kaachupetti eshtu ide",
        ],
        "add_stock": [
            "10 kg akki haaku", "sakkare stock jaasti maadi",
            "haalu order maadi", "bele taa", "enne tin bandide haaku",
            "hittu 5 kg haaku", "sabbu order maadi", "tuppa jaasti maadi",
            "chai taa", "sakkare mugiyutte ide order maadi",
            "benne hosa stock haaku", "20 kg godhi kharidi",
            "biscuit taa", "coffee stock jaasti maadi",
            "tarkari stock jaasti maadi", "akki order kodu",
            "50 noodle packet haaku", "hosa maal bandide haaku",
            "delivery bandide update maadi", "hosa stock haaku system alli",
            "masala thumbi", "100 sabbu haaku", "uppu 10 kg jaasti maadi",
            "bread stock haaku", "motte taa",
            "5 litre enne tin haaku", "juice jaasti maadi",
            "cool drink order maadi", "masala order maadi",
            "bandha maal system alli haaku", "hosa maal haaku",
            "habba ge munche stock jaasti maadi", "weekend ge mundhe stock jaasti",
            "25 kg hittu order maadi", "kapada thovuva powder haaku",
            "10 kg kadala bele haaku", "jaldi restock maadi",
            "hosa batch haaku", "delivery nantara update maadi",
            "saamaan bandide add maadi",
        ],
        "get_prediction": [
            "naale akki eshtu biku", "munde vaara sakkare bedu eshtu",
            "bele forecast heli", "haalu bedu eshtu aagutte",
            "enne upayoga eshtu aagutte", "naale ge hittu eshtu beku",
            "bedu andaaja beku", "munde vaara eshtu order maadali",
            "habba alli eshtu biku", "weekend alli enu jasti biku",
            "naale best seller yaav", "somvara ge forecast heli",
            "sakkare bedu andaaja haaki", "munde tiNgaLige stock plan",
            "yaav item jasti biku heli", "7 dina ge stock eshtu irabeku",
            "akki biku andaaja kodu", "ee vaara enu aaagabeku",
            "jasti bikuna item heli", "raje dinakke forecast",
            "naale hisaab haaki", "ee vaara enne eshtu beku",
            "ella item bedu heli", "nidhaana item heli",
            "bhanuvaara enu beku aagutte",
        ],
        "low_stock_alert": [
            "yaav item kama aagutte", "low stock pattiyaa kodu",
            "enu mugiyutte ide", "minimum ge keLage yaav ide",
            "stock mugiyutte ide yaav", "jaldi order beku yaav item ge",
            "khaali aagutte ide alert kodu", "khaali shelf yaav",
            "urgent order yaav item ge", "naale yaav mugiyutte",
            "stock warning kodu", "yaav maal mugitu",
            "yaav item thumbaa kama ide", "naale reorder yaav",
            "zero stock yaav ide", "emergency alli yaav order maadali",
            "jaldi mugiyutte ide yaav", "pratidinada low stock report",
            "almost mugiyitu enu", "yaav maal illa",
            "naale eshtu mugitu", "stock alert kodu",
            "indina shortage heli", "yaav urgent ide",
            "aLavo kama stock yaav",
        ],
        "help": [
            "hege upayogisu", "nange sahaya beku", "guide tilikoli",
            "stock hege check maadali", "tutorial beku", "enu maadabahudu",
            "item hege haakaali", "stock add maadalu heli",
            "forecast hege kaela yuttade", "prediction arthavenu",
            "upayoga vidhana heli", "praarambbha hege maadaali",
            "yaav features ide", "help menu tilikoli", "enu maadabahude",
            "minimum stock hege set maadali", "report hege nodali",
            "alert hege set maadali", "inventory update hege maadali",
            "hosa nanu hege praarambbisu", "app hege chaalaisu",
            "stock prediction endare enu", "sahaya beku",
            "smartstock endare enu", "hege start maadali",
        ],
    },

    # ──────────────── CODE-MIXED (Hindi-English + Kannada-English) ────────────────
    "Code-Mixed": {
        "check_stock": [
            "rice ka stock check karo please", "sugar kitna hai currently",
            "dal stock kya show karega", "milk available hai kya today",
            "soap inventory check please", "oil remaining kitna hai abhi",
            "atta stock level kya hai", "tea kitna stock hai tell me",
            "sab items check kar do", "current stock of chawal batao",
            "akki stock eshtu ide check karo", "sakkare eshtu remaining",
            "bele stock konchaa ide check", "sabbu eshtu ide tell me",
            "enne stock remaining eshtu", "chai powder stock kitna ide",
            "sab ka inventory status do", "complete stock report please",
            "which items are low batao", "kya available hai today",
            "stock summary abhi do", "quick inventory check maadi",
        ],
        "add_stock": [
            "rice ka 10 kg add karo please", "sugar restock karo urgent",
            "dal order maadi jaldi", "milk stock add karo",
            "enne 5 litre haaku stock mein", "hittu 25 kg order karo",
            "sabbu 100 haaku please", "tuppada stock jaasti karo",
            "chai ka naya stock daalo", "stock update karo after delivery",
            "naya maal add karo system mein", "haalu order maadi please",
            "bele 10 kg add please", "restock sab items karo",
        ],
        "get_prediction": [
            "kal kitna rice bikega predict karo", "sugar demand forecast please",
            "naale akki sales prediction do", "next week ka demand batao",
            "weekend mein kya jyada bikega forecast", "demand estimate please batao",
            "stock prediction for next 7 days", "naale ge forecast maadi please",
            "kya khareedna chahiye next week ke liye", "predict sales for tomorrow please",
        ],
        "low_stock_alert": [
            "stock low hai kya aaj", "alert do low items ke liye",
            "kya khatam ho raha hai tell me", "low stock items list please",
            "yaav items low ide batao", "urgent items kya hain",
            "stock warning do abhi", "khaali hone wale items batao",
            "emergency restock kaun sa", "aaj kya order karna padega",
        ],
        "help": [
            "help chahiye kaise use karu", "how to add stock batao",
            "system kaise kaam karta hai explain", "guide please dikhao",
            "smartstock kya hai aur kaise use kare", "hege use maadali please tell me",
            "tutorial chahiye help karo", "naye user ke liye guide",
            "features kya hain explain karo", "getting started please help",
        ],
    },
}


def expand_sentences(sentences_dict, target_per_class=80):
    """Expand each intent's sentence list to at least `target_per_class` entries.

    Fills up to the target by cycling through the base list with light
    variations (appending common suffixes) to reach the required count.

    Args:
        sentences_dict: Dict mapping intent → list of sentences.
        target_per_class: Minimum sentences per intent after expansion.

    Returns:
        Expanded dict with the same keys.
    """
    suffixes = [
        "", " please", " now", " quickly", " today", " asap",
        " jaldi", " abhi", " please tell me", " batao",
    ]
    expanded = {}
    for intent, sents in sentences_dict.items():
        result = list(sents)
        idx = 0
        while len(result) < target_per_class:
            base = sents[idx % len(sents)]
            suffix = suffixes[(idx // len(sents)) % len(suffixes)]
            candidate = (base + suffix).strip()
            if candidate not in result:
                result.append(candidate)
            idx += 1
        expanded[intent] = result
    return expanded


def build_dataframe():
    """Build the full multilingual NLP dataset DataFrame.

    Returns:
        DataFrame with columns: sentence, language, intent, product_mentioned.
    """
    # Common product keywords for a lightweight product extraction heuristic
    product_keywords = {
        "rice": ["rice", "chawal", "akki"],
        "sugar": ["sugar", "cheeni", "sakkare"],
        "dal": ["dal", "bele"],
        "milk": ["milk", "doodh", "haalu"],
        "oil": ["oil", "tel", "enne"],
        "flour": ["flour", "atta", "hittu", "maida"],
        "tea": ["tea", "chai"],
        "soap": ["soap", "sabun", "sabbu"],
        "ghee": ["ghee", "tuppa", "tuppada"],
        "salt": ["salt", "namak", "uppu"],
    }

    rows = []
    for language, intents in SENTENCES.items():
        intents = expand_sentences(intents, target_per_class=80)
        for intent, sents in intents.items():
            for sent in sents:
                sent_lower = sent.lower()
                product = "unknown"
                for prod, keywords in product_keywords.items():
                    if any(kw in sent_lower for kw in keywords):
                        product = prod
                        break
                rows.append({
                    "sentence": sent.strip(),
                    "language": language,
                    "intent": intent,
                    "product_mentioned": product
                })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def print_statistics(df):
    """Print counts per language, per intent, and overall total."""
    print("\n  === Dataset Statistics ===")
    print(f"  Total sentences : {len(df)}")
    print("\n  By Language:")
    for lang, cnt in df["language"].value_counts().items():
        print(f"    {lang:15s} : {cnt}")
    print("\n  By Intent:")
    for intent, cnt in df["intent"].value_counts().items():
        print(f"    {intent:20s} : {cnt}")
    print("\n  Language × Intent cross-tab:")
    print(pd.crosstab(df["language"], df["intent"]).to_string())


def main():
    print("=" * 60)
    print("STEP 8: NLP Dataset Creation")
    print("=" * 60)

    print("\n  [1/2] Building multilingual sentence corpus...")
    df = build_dataframe()

    out_path = os.path.join(NLP_DIR, "smartstock_nlp_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    print_statistics(df)
    print("\n  [DONE] NLP dataset created.")


if __name__ == "__main__":
    main()
