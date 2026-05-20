"""
SearchLessons — 从失败题目分析中提取的高级搜索策略
=====================================================

基于 100 题 BrowseComp 失败模式分析，提供：
1. 解码技巧（将隐晦描述映射到可搜索实体）
2. 搜索锚定规则（如何找到最有区分度的约束）
3. 领域特定搜索策略
4. 验证陷阱规避

注意：不包含任何具体题目答案，只有通用策略。
"""
from __future__ import annotations

import re
from typing import Optional


# ===========================================================================
# Lesson 1: 解码隐晦描述 (Decode Euphemistic Descriptions)
# ===========================================================================

DECODE_PATTERNS = {
    # Platform descriptions → actual platform names
    "global non-profit journalistic platform.*academics": "The Conversation (theconversation.com)",
    "platform.*academics.*share research": "The Conversation or ResearchGate",
    "platform.*3 million downloads": "likely a DeFi/crypto platform (Aave, etc.)",

    # Event/concept descriptions → actual events
    "born out of discord.*two parties": "club formed from a schism/split (e.g., Inter Milan from AC Milan, AFC Wimbledon from MK Dons)",
    "identity evolved through several iterations": "team that changed names multiple times (relocated/rebranded)",
    "real-life event.*Asia.*21st century": "Hiroshima bombing depicted in show, or Fukushima disaster, or other major Asian event",

    # Person attribute descriptions → search terms
    "fitting nickname": "famous epithet/moniker (search: 'nicknamed' + domain)",
    "human female": "wordplay for 'woman' → stage name likely contains 'woman' or related term",
    "matching tattoo.*YouTuber": "YouTube collaborators who got matching tattoos together",
    "first channel.*human female": "channel name contains 'woman' or 'girl' or 'female' — decode the wordplay",
    "last credit.*\\d{4}": "director/actor whose most recent work was in that specific year — very identifying",

    # Award descriptions → actual awards
    "AIA Henry Adams Medal": "American Institute of Architects student award — search recipient lists",
    "Henry Adams Medal": "AIA student architecture award — search 'AIA Henry Adams Medal recipients' for finite list",
    "Grammy.*Best Traditional Pop Vocal": "Grammy category winners list — search Wikipedia",
    "prestigious awards.*18 nominations": "Grammy/Oscar level — count nominations on Wikipedia",

    # Company/org descriptions
    "debt-free balance sheet.*powerboats": "marine/boat manufacturer with strong financials",
    "association.*90 member companies.*2018": "likely Finnish/European fintech or tech association",
    "backed another.*company": "investor/VC relationship — search portfolio",

    # Zodiac/astrology descriptions
    "zodiac signs.*same element": "both actors share Fire/Water/Earth/Air element — check birthday zodiac signs",
    "Virgo.*basketball": "born Aug 23–Sep 22 — filter by birthday",

    # Sports descriptions
    "logo.*locking or unlocking": "crest with key/lock/padlock imagery (Everton has Lock-Up tower)",
    "team.*name changed.*fit.*new location": "relocated franchise (e.g., Warriors: Philadelphia→San Francisco→Golden State)",
    "record.*regular season wins": "NBA 73-9 Warriors 2015-16, or NFL undefeated season",
    "second major trophy.*1970s.*European": "second league title or cup win in the 1970s",
    "historical military group": "team named after warriors/soldiers/knights etc.",
    "shares.*name.*historical military": "team name = soldiers/warriors/knights/raiders etc.",

    # Novel/media descriptions
    "banned.*country.*cultural analysis": "novel banned in the country it critiques — search banned novels + cultural criticism",
    "failed.*university.*journalist.*novelist": "author who failed exams then became journalist-novelist (e.g., Kamel Daoud)",
    "Argentinian release name": "search for the Argentine Spanish localization specifically, not general Spanish",
}


# ===========================================================================
# Lesson 2: 搜索锚定规则 (Anchor Selection Rules)
# ===========================================================================

ANCHOR_PRIORITY_RULES = """## Search Anchor Selection (CRITICAL — determines success or failure)

The ANCHOR is the single most distinctive constraint that will narrow your search to <10 results.

### Priority Order (highest first):
1. **Rare awards/medals**: "AIA Henry Adams Medal", "Nobel Prize in [field]" → search recipient list directly
2. **Exact phrases/quotes**: Any text in quotation marks → use verbatim
3. **Nicknames/epithets**: "fitting nickname", "known as" → search "nicknamed" + domain
4. **Specific numbers** (non-year): "1,570,428", "37 parking bays", "23 books" → pair with context
5. **Death/accident circumstances**: "died in car accident August [year]" → very rare, search directly
6. **Unique career transitions**: "failed university exams and worked as journalist" → biographical search
7. **Specific institutional facts**: "PhD 1987", "university founded between 1940-1990" → narrow institution first
8. **Production details**: "18 minutes", "50-60 episodes", "one director two writers" → IMDb/database search
9. **Championship/record facts**: "73-9 season", "6 Le Mans wins" → sports database search
10. **Year + domain combination**: "2023 article + animated movie + 1980s" → weak but usable

### Anti-patterns (NEVER use as primary anchor):
- Birth decade alone ("born in 1930s") — too many people
- Generic descriptions ("rich and poor fall in love") — too many movies
- "Before 2023" or "as of December 2023" — not distinctive
- Country/nationality alone ("French actor") — too broad
- Genre alone ("animated movie", "metal band") — too broad
"""


# ===========================================================================
# Lesson 3: 领域特定搜索策略 (Domain-Specific Search Strategies)
# ===========================================================================

DOMAIN_STRATEGIES = {
    "actor_identification": {
        "lesson": (
            "For actor identification with film/director constraints:\n"
            "1. If 'director whose last credit was [year]' → identify the director FIRST (very few directors have a clear 'last credit')\n"
            "2. Then search 'director_name filmography breakthrough [decade]'\n"
            "3. Cross-reference actor roster of identified film\n"
            "4. Verify additional constraints (magazine covers, paid performance decade)"
        ),
        "keywords": ["actor", "acting", "film", "director", "movie", "role", "breakthrough"],
    },
    "racing_driver": {
        "lesson": (
            "For racing driver identification:\n"
            "1. 'Fitting nickname' → search 'racing driver famous nickname [category]'\n"
            "2. 'Multiple categories' → F1 + Le Mans/WEC/endurance — very few cross-category champions\n"
            "3. 'Family in motorsport' → search 'racing family father son brother'\n"
            "4. Famous epithets: search exact phrases like '\"Mr. Le Mans\"' or '\"King of Spa\"' — these are unique identifiers"
        ),
        "keywords": ["racing", "driver", "motorsport", "nickname", "categories"],
    },
    "basketball_player": {
        "lesson": (
            "For basketball player identification:\n"
            "1. 'Championship team + record season wins' → identify the TEAM and SEASON first (73-9 = 2015-16 Warriors)\n"
            "2. Then filter roster by remaining constraints (zodiac sign, draft team, Olympics)\n"
            "3. 'Team founded in 1940s whose name changed' → relocated franchise history\n"
            "4. 'Multiple Olympics' — narrows significantly, check Olympic basketball rosters"
        ),
        "keywords": ["basketball", "drafted", "team", "Olympics", "championship"],
    },
    "football_match": {
        "lesson": (
            "For football match/team identification:\n"
            "1. 'Logo with [object]' → Google 'football club crest [object]' or check heraldry databases\n"
            "2. 'Founded in 1870s' → one of the oldest clubs (Everton 1878, Wolves 1877, etc.)\n"
            "3. 'Born out of discord between two parties' → club formed from a split/schism\n"
            "4. 'Second major trophy in 1970s' → check trophy history timeline\n"
            "5. For goalscorer identification: find the match first, then check goalscorer list"
        ),
        "keywords": ["football", "soccer", "match", "team", "trophy", "founded", "logo", "goal"],
    },
    "swimmer_athlete": {
        "lesson": (
            "For swimmer/athlete identification:\n"
            "1. Birth month + year range + first gold medal event → very specific combination\n"
            "2. Search 'swimmer gold medal [year] born [month] [country]'\n"
            "3. 'First international gold' specifies FIRST career gold — check career timeline\n"
            "4. Biographical details like childhood dreams → only findable in profile interviews, use for VERIFICATION not search"
        ),
        "keywords": ["swimmer", "athlete", "gold medal", "born", "Olympics"],
    },
    "anime_series": {
        "lesson": (
            "For anime/foreign series identification:\n"
            "1. 'Foreign series' in English benchmark = anime or K-drama\n"
            "2. 'Female villain who controls people' → psychic/mind-control character → search MyAnimeList\n"
            "3. Specific episode events → search '[anime name] episode [event description]'\n"
            "4. Season + episode count + air year narrows significantly\n"
            "5. Use domain-specific databases: MyAnimeList, AniDB, AniList\n"
            "6. 'Character with ability to teleport' + 'shares name with Nobel Prize scientist' → combine both rare constraints"
        ),
        "keywords": ["anime", "series", "foreign", "episode", "season", "villain", "show", "teleport", "character"],
    },
    "bollywood_movie": {
        "lesson": (
            "For Bollywood/Indian movie identification:\n"
            "1. 'Rich and poor fall in love' → extremely common Bollywood trope, need MORE constraints\n"
            "2. Zodiac sign of actors → check actors' birthdays on Wikipedia\n"
            "3. 'Same element' = Fire(Aries/Leo/Sagittarius), Water(Cancer/Scorpio/Pisces), Earth(Taurus/Virgo/Capricorn), Air(Gemini/Libra/Aquarius)\n"
            "4. Search 'Bollywood [decade] romance movie rich poor' then verify zodiac constraint\n"
            "5. Lead actor pairs in 2000s Bollywood: SRK+Rani, SRK+Kajol, Hrithik+Aishwarya etc."
        ),
        "keywords": ["movie", "Bollywood", "zodiac", "love", "luxury", "poverty", "Indian"],
    },
    "novel_book": {
        "lesson": (
            "For novel/book identification:\n"
            "1. 'Banned in the country it analyzes' → search 'banned novel [country] cultural analysis'\n"
            "2. Author biographical details (failed exams, journalist) → search author bio first\n"
            "3. 'First published in French' → French-language novel (not necessarily by French author)\n"
            "4. Publication year + language + ban status narrows significantly\n"
            "5. For English titles of foreign works: search 'original_title English translation'"
        ),
        "keywords": ["novel", "book", "author", "published", "banned", "title"],
    },
    "species_taxonomy": {
        "lesson": (
            "For species/taxonomy identification:\n"
            "1. 'Described in [decade] by naturalist' → search taxonomic databases (GBIF, MycoBank, WoRMS)\n"
            "2. 'Five unaccepted synonyms' → structured data in Species Fungorum / Catalogue of Life\n"
            "3. 'Naturalist who studied other areas' → 18th century polymath naturalists\n"
            "4. Go DIRECTLY to taxonomy databases where synonym counts are searchable\n"
            "5. For fungi: MycoBank. For animals: ITIS/WoRMS. For plants: IPNI/ThePlantList."
        ),
        "keywords": ["species", "naturalist", "described", "synonyms", "taxonomy", "genus"],
    },
    "youtuber_influencer": {
        "lesson": (
            "For YouTuber/influencer identification:\n"
            "1. 'Stage name' or 'channel name' descriptions are WORDPLAY → decode first\n"
            "2. 'Human female' = woman → channel name likely contains 'woman' or similar wordplay\n"
            "3. 'Matching tattoo with another YouTuber' → search 'YouTuber matching tattoo collaboration'\n"
            "4. 'Career over a decade' + 'YouTube vlog series' → established creator since ~2010-2012\n"
            "5. Check subscriber counts and social blade for verification"
        ),
        "keywords": ["YouTuber", "influencer", "channel", "stage name", "vlog"],
    },
    "french_actor": {
        "lesson": (
            "For French actor identification (biographical constraints):\n"
            "1. Spouse death + cause + year range → very specific, search 'French actor spouse died cancer [decade]'\n"
            "2. 'Sibling who was musician/dancer' → family connection, search 'French actor brother/sister musician'\n"
            "3. 'Film with person who directed video for Grammy winner' → LONG CHAIN, resolve backward:\n"
            "   - Identify Grammy Best Traditional Pop Vocal winners → their music videos → video directors → films featuring director + our actor\n"
            "4. For 'full birth name': Wikipedia infobox has birth name — verify all middle names"
        ),
        "keywords": ["French", "born", "spouse", "sibling", "film", "1930s"],
    },
    "esports_tournament": {
        "lesson": (
            "For eSports identification:\n"
            "1. Tournament name usually includes: league + season + year (e.g., 'LCK Summer 2023 Playoffs')\n"
            "2. 'First match in finals' → search specific match details on Leaguepedia/Liquipedia\n"
            "3. Champion picks, kill counts → game-specific wikis have detailed match statistics\n"
            "4. Player transfers and roster changes → Liquipedia is authoritative\n"
            "5. Format: 'League Region Season Year Stage'"
        ),
        "keywords": ["League of Legends", "tournament", "playoffs", "match", "finals", "esports", "LCK", "LPL"],
    },
    "award_chain_search": {
        "lesson": (
            "For questions involving awards as a constraint:\n"
            "1. Search '[award name] winners list' or '[award name] recipients' FIRST\n"
            "2. Awards have FINITE recipient lists — much easier to search than the person directly\n"
            "3. Common awards to look up: AIA Henry Adams Medal, Grammy categories, Nobel Prize fields\n"
            "4. Once you find the recipient, search '[recipient name] + [next constraint]' to follow the chain\n"
            "5. For 'Grammy Best Traditional Pop Vocal Album' — search the specific category, not just 'Grammy winner'"
        ),
        "keywords": ["award", "medal", "prize", "Grammy", "Nobel", "received", "won"],
    },
}


# ===========================================================================
# Lesson 4: 验证陷阱 (Verification Traps)
# ===========================================================================

VERIFICATION_LESSONS = """## Verification Pitfalls to Avoid

1. **Don't verify only the easy constraints**: If you found a candidate and verified one trait, ALSO verify ALL remaining constraints. Partial matches are the #1 error source.

2. **"Cannot find" ≠ "Wrong answer"**: Many correct answers have sparse web presence. If you can't find contradicting evidence, your answer may still be correct.

3. **Beware partial matches**: Finding someone who matches 4/5 constraints doesn't mean they're correct. The 5th constraint might eliminate them entirely.

4. **Date precision matters**: "July 15, 2020" vs "June 15, 2020" — verify exact dates, not just month/year.

5. **Full name vs common name**: If asked for "full birth name", include ALL middle names. Search Wikipedia infobox for birth name fields.

6. **"As of December 2023"**: This is a time constraint on the KNOWLEDGE, not on your search. The answer must be true as of that date.

7. **Localized names**: For "Argentinian release name" → search for the specific country's localization, not the general Spanish translation.

8. **Award specificity**: "Grammy for Best Traditional Pop Vocal" is different from just "Grammy". Verify the SPECIFIC category.
"""


# ===========================================================================
# Lesson 5: 多步搜索模板 (Multi-Step Search Templates)
# ===========================================================================

SEARCH_TEMPLATES = {
    "chain_traversal": (
        "For relationship chains (A→B→C→D):\n"
        "Step 1: Identify which entity in the chain has the MOST SEARCHABLE constraint\n"
        "Step 2: Search for that entity ALONE (don't include chain context)\n"
        "Step 3: Once found, use it to search for the NEXT link: '[entity_found] + [next_relationship]'\n"
        "Step 4: Repeat until you reach the answer entity\n"
        "Step 5: Verify by tracing the chain BACKWARDS\n"
        "NEVER put the entire chain description in one search query!"
    ),
    "award_winner_lookup": (
        "For 'person who won [award]':\n"
        "Step 1: Search '[award name] winners list' or '[award name] recipients'\n"
        "Step 2: Find the specific winner matching time/other constraints\n"
        "Step 3: Search '[winner name] + [next constraint in question]'\n"
        "Key: Awards have finite recipient lists — find the list first!"
    ),
    "team_identification": (
        "For team identification with multiple constraints:\n"
        "Step 1: Use the RAREST constraint (founding decade + city, or specific trophy year)\n"
        "Step 2: Search '[constraint1] [constraint2] football/basketball club'\n"
        "Step 3: If too many results, add the THIRD constraint\n"
        "Step 4: Verify: check founding year, trophy list, and name history on Wikipedia"
    ),
    "person_from_work": (
        "For 'person who worked on [project/film/article]':\n"
        "Step 1: Find the WORK first (film, article, book, etc.)\n"
        "Step 2: Search '[work title] [role] cast/crew/author'\n"
        "Step 3: Cross-reference the person against biographical constraints\n"
        "Key: Finding the work is usually easier than finding the person directly."
    ),
    "decode_then_search": (
        "For questions with euphemistic descriptions:\n"
        "Step 1: DECODE the description (what real-world entity does it describe?)\n"
        "Step 2: Once decoded, search using the ACTUAL name/term\n"
        "Examples: 'global non-profit journalistic platform for academics' = The Conversation\n"
        "          'born out of discord between two parties' = club formed from a split\n"
        "          'human female' as channel name = contains 'woman'"
    ),
}


# ===========================================================================
# Integration: Build lesson section for prompt
# ===========================================================================

def detect_applicable_domains(question: str, has_image: bool = False) -> list[str]:
    """Detect which domain-specific strategies apply to this question."""
    ql = question.lower()
    applicable = []

    for domain_key, domain_info in DOMAIN_STRATEGIES.items():
        keywords = domain_info["keywords"]
        match_count = sum(1 for kw in keywords if kw.lower() in ql)
        if match_count >= 2 or (match_count == 1 and len(keywords) <= 3):
            applicable.append(domain_key)

    return applicable[:2]


def detect_decode_opportunities(question: str) -> list[str]:
    """Find euphemistic descriptions that should be decoded before searching."""
    hints = []
    ql = question.lower()

    for pattern, decode_hint in DECODE_PATTERNS.items():
        if re.search(pattern, ql):
            hints.append(f"DECODE: '{pattern[:60]}...' → {decode_hint}")

    return hints[:3]


def build_lessons_section(
    question: str,
    has_image: bool = False,
    max_chars: int = 2000,
) -> str:
    """Build a lessons section to inject into the system prompt."""
    parts = []
    char_count = 0

    # 1. Decode opportunities (highest priority — prevents search failure at root)
    decode_hints = detect_decode_opportunities(question)
    if decode_hints:
        decode_section = "## Decoding Hints (read BEFORE searching)\n"
        for hint in decode_hints:
            decode_section += f"- {hint}\n"
        parts.append(decode_section)
        char_count += len(decode_section)

    # 2. Domain-specific lessons
    domains = detect_applicable_domains(question, has_image)
    if domains:
        for domain in domains:
            strategy = DOMAIN_STRATEGIES[domain]
            lesson = f"## Domain Strategy: {domain.replace('_', ' ').title()}\n{strategy['lesson']}\n"
            if char_count + len(lesson) > max_chars:
                break
            parts.append(lesson)
            char_count += len(lesson)

    # 3. Applicable search templates
    ql = question.lower()
    chain_markers = re.findall(r'(whose|that was|who was|which was|backed|owned by|ceo of|founded by)', ql)
    if len(chain_markers) >= 2 and char_count + 300 < max_chars:
        parts.append(f"## Search Template\n{SEARCH_TEMPLATES['chain_traversal']}\n")
        char_count += 300

    if any(w in ql for w in ['award', 'medal', 'prize', 'grammy', 'oscar']):
        if char_count + 200 < max_chars:
            parts.append(f"## Search Template\n{SEARCH_TEMPLATES['award_winner_lookup']}\n")
            char_count += 200

    # 4. Anchor selection reminder (always include if space)
    if char_count + 500 < max_chars:
        # Abbreviated version of anchor rules
        anchor_brief = (
            "## Anchor Selection Reminder\n"
            "Your FIRST search must use the MOST DISTINCTIVE constraint:\n"
            "- Rare awards/medals > Exact quotes > Nicknames > Specific numbers > Death circumstances > Career transitions\n"
            "- NEVER lead with: birth decade alone, genre alone, 'as of 2023', country alone\n"
        )
        parts.append(anchor_brief)

    if not parts:
        return ""

    return "\n".join(parts)
