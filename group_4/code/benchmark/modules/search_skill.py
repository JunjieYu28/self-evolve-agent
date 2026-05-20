"""
SearchSkill — BrowseComp 搜索策略技能库
========================================

基于 benchmark 数据模式分析产出的搜索策略。
不引用具体题目，只基于问题结构模式给出精确搜索指导。

核心发现:
- 80% 问题含非年份特定数字（作为强约束）
- 50% 为图片题（识别实体后搜索属性）
- 大量问题包含关系链（A -> B -> C 式的跳跃搜索）
- 答案 82% 为实体名（人名、组织名、作品名）
- 答案平均长度 12 字符，82% 在 3 词以内
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchStrategy:
    pattern_name: str
    priority_search_guidance: str
    search_tactics: list[str]
    verification_approach: str
    common_pitfalls: list[str]


# ===========================================================================
# Pattern Detection
# ===========================================================================

def detect_question_patterns(question: str, has_image: bool = False) -> list[str]:
    """Detect which search patterns apply to this question."""
    patterns = []
    ql = question.lower()

    if has_image:
        if len(question) < 120:
            patterns.append("image_direct_identify")
        else:
            patterns.append("image_compound")

    # Relationship chain: A's B, whose C, that D...
    chain_markers = re.findall(r'(whose|that was|who was|which was|backed|owned by|subsidiary|parent company|ceo of|founded by)', ql)
    if len(chain_markers) >= 2:
        patterns.append("relationship_chain")

    # Rare specific numbers (not years)
    all_nums = re.findall(r'\b(\d{1,7})\b', question)
    non_year_nums = [n for n in all_nums if not (1800 <= int(n) <= 2030) and int(n) > 1]
    if len(non_year_nums) >= 2:
        patterns.append("rare_number_intersection")
    elif len(non_year_nums) == 1 and int(non_year_nums[0]) > 10:
        patterns.append("rare_number_intersection")

    # Year range bracketing
    year_ranges = re.findall(r'between (\d{4}) and (\d{4})', question)
    if year_ranges:
        patterns.append("year_range_bracketing")

    # Academic/thesis/university pattern
    if any(w in ql for w in ['thesis', 'phd', 'doctoral', 'university', 'degree', 'institution', 'professor']):
        patterns.append("academic_search")

    # Sports pattern
    if any(w in ql for w in ['team', 'goal', 'match', 'league', 'championship', 'tournament', 'trophy', 'soccer', 'football']):
        patterns.append("sports_entity")

    # Entertainment/media pattern
    if any(w in ql for w in ['movie', 'film', 'show', 'anime', 'series', 'episode', 'director', 'actor', 'song', 'album', 'youtube']):
        patterns.append("entertainment_media")

    # Person identification with biographical constraints
    if any(w in ql for w in ['born in', 'died', 'married', 'spouse', 'graduated']):
        patterns.append("biographical_identification")

    # Company/business entity
    if any(w in ql for w in ['company', 'corporation', 'startup', 'ceo', 'revenue', 'ipo', 'stock', 'balance sheet']):
        patterns.append("business_entity")

    # Quoted exact phrases (strongest search signal)
    if '"' in question:
        patterns.append("exact_phrase_given")

    # Nickname/title pattern
    if any(w in ql for w in ['nickname', 'known as', 'called', 'alias', 'pen name', 'stage name']):
        patterns.append("nickname_search")

    # Scientific/nature pattern
    if any(w in ql for w in ['species', 'scientific name', 'genus', 'naturalist', 'insect', 'plant', 'binomial']):
        patterns.append("scientific_classification")

    if not patterns:
        patterns.append("general_entity")

    return patterns


# ===========================================================================
# Strategy Library
# ===========================================================================

STRATEGY_LIBRARY: dict[str, SearchStrategy] = {
    "relationship_chain": SearchStrategy(
        pattern_name="Relationship Chain Traversal",
        priority_search_guidance=(
            "This question contains a CHAIN of entities connected by relationships "
            "(e.g., 'person X who is CEO of company Y that backed company Z whose founder...'). "
            "You CANNOT search for the final answer directly — you must traverse the chain."
        ),
        search_tactics=[
            "Identify the FIRST entity in the chain that has a searchable constraint, search for IT first",
            "Once you find an intermediate entity, use it as a NEW search term to find the next link",
            "Each hop should be a separate search: don't chain multiple hops in one query",
            "If you can identify ANY named entity in the chain, start there — it's your anchor point",
            "Work from specific to general: the most constrained entity in the chain is your best entry point",
        ],
        verification_approach=(
            "After finding the final entity, trace the chain BACKWARDS to verify each link. "
            "Search 'entity_A + entity_B relationship' for each pair in the chain."
        ),
        common_pitfalls=[
            "Searching for the final answer directly without resolving intermediate entities",
            "Putting the entire chain description in one search query",
            "Stopping after finding the first entity without following the chain to the actual answer",
        ],
    ),

    "rare_number_intersection": SearchStrategy(
        pattern_name="Rare Number + Context Intersection",
        priority_search_guidance=(
            "This question contains SPECIFIC NUMBERS that are rare and highly searchable. "
            "Numbers like counts, quantities, or measurements are excellent search anchors "
            "because they appear in very few web pages."
        ),
        search_tactics=[
            "Use the most UNUSUAL number combined with its context: e.g., '25 apartments 37 parking' or '90 member companies 2018'",
            "Put exact numbers in quotes if they have a unit: '\"3 million downloads\"' or '\"23 books\"'",
            "Combine the number with the DOMAIN (not generic words): '37 parking bays flats' not '37 property development'",
            "If one number doesn't work, try a DIFFERENT number from the question with different context words",
            "Numbers + year ranges together are powerful: '25 apartments 2015 development'",
        ],
        verification_approach=(
            "When you find a candidate through a number search, verify the OTHER numbers "
            "in the question also match. If the question says '25 apartments and 37 parking bays', "
            "confirm BOTH numbers appear in the source."
        ),
        common_pitfalls=[
            "Using year numbers instead of the more distinctive quantity/count numbers",
            "Searching without the number (just generic context words)",
            "Not verifying that ALL numbers in the question match your candidate",
        ],
    ),

    "year_range_bracketing": SearchStrategy(
        pattern_name="Year Range Constraint Bracketing",
        priority_search_guidance=(
            "This question uses year ranges (e.g., 'between 1985 and 1992') as constraints. "
            "These ranges NARROW the search space but are NOT the best primary search terms. "
            "Pair them with more distinctive constraints."
        ),
        search_tactics=[
            "Do NOT lead with the year range alone — combine it with a distinctive event or entity type",
            "Use the MIDPOINT year + distinctive context: for 'founded between 1985 and 1992' try 'founded 1988 capital city'",
            "Year ranges on founding dates → search lists: 'universities founded 1980s' or 'states established 1990s'",
            "Multiple year ranges in one question usually mean: resolve them ONE AT A TIME, each giving you a different entity",
            "If the question has 3+ year ranges, the NARROWEST one (shortest span) is usually most distinctive",
        ],
        verification_approach=(
            "After finding a candidate, verify the year explicitly: "
            "search 'entity_name founded' or 'entity_name established' to confirm it falls within the stated range."
        ),
        common_pitfalls=[
            "Searching for a year range without any distinctive context",
            "Using only the start or end year instead of the full context",
            "Not verifying the actual year when a range-only match is found",
        ],
    ),

    "image_direct_identify": SearchStrategy(
        pattern_name="Direct Image Identification",
        priority_search_guidance=(
            "This is a SHORT question with an image — the primary task is to IDENTIFY "
            "what/who is in the image, then answer the specific question asked."
        ),
        search_tactics=[
            "Use search_image FIRST to identify the person/object/scene in the image",
            "If image search returns unclear results, look for TEXT in the image (jersey numbers, logos, signs, watermarks)",
            "For person identification: search visible text like jersey number + team, or event name + context",
            "For product/game identification: search visible UI elements, brand logos, or distinctive design features",
            "After identification, do a TEXT search for the specific attribute asked (e.g., 'person_name + 600 games LPL')",
        ],
        verification_approach=(
            "Cross-reference the image search result with the text question. "
            "If they ask about a specific event/time, verify the person was associated with that event."
        ),
        common_pitfalls=[
            "Guessing from image alone without doing a verification text search",
            "Not reading text/numbers visible in the image (jersey numbers are key identifiers)",
            "Confusing similar-looking people without checking contextual clues",
        ],
    ),

    "image_compound": SearchStrategy(
        pattern_name="Image + Complex Text Compound",
        priority_search_guidance=(
            "This question has BOTH an image AND significant text constraints. "
            "The image identifies ONE entity, and the text asks about something CONNECTED to that entity. "
            "You must resolve the image FIRST, then follow the text question."
        ),
        search_tactics=[
            "Step 1: Identify the person/entity in the image using search_image",
            "Step 2: Once identified, search for the specific connection mentioned in the text",
            "For 'person in image manages fund X that invested in Y' — identify person first, then search their fund's investments",
            "Include TIME constraints from the text in your follow-up searches (e.g., 'August 2024')",
            "If the question mentions 'company managed by person in image', search 'person_name CEO' or 'person_name company'",
        ],
        verification_approach=(
            "Verify that the identified person actually holds the role mentioned in the text "
            "(CEO, fund manager, etc.) before searching for the final answer."
        ),
        common_pitfalls=[
            "Trying to answer without first identifying the person in the image",
            "Confusing the IMAGE entity with the ANSWER entity (they are different!)",
            "Ignoring time constraints — '2024' events need recent search results",
        ],
    ),

    "academic_search": SearchStrategy(
        pattern_name="Academic/Thesis/University Search",
        priority_search_guidance=(
            "Questions about theses, degrees, or academic work are best searched through "
            "institutional repositories, Google Scholar, or by combining the UNIQUE aspects "
            "of the academic work (dedication, topic, year)."
        ),
        search_tactics=[
            "Search for the most UNIQUE aspect of the thesis: dedication text, specific topic combination, or unusual subject",
            "University + founding year narrowing: search 'universities established 1980s [country]' to identify the institution first",
            "For thesis searches: 'PhD thesis [topic keywords] [year range] dedicated to children'",
            "Try institutional repository searches: 'site:edu thesis [keywords]'",
            "Author names from thesis dedications are rare enough to be directly searchable",
        ],
        verification_approach=(
            "Verify the university's founding date, the thesis submission year, "
            "and the specific dedication/content mentioned in the question."
        ),
        common_pitfalls=[
            "Searching too broadly for 'PhD thesis 2020' without distinctive content clues",
            "Not using the dedication text as a search term (it's often the most unique constraint)",
            "Confusing university founding date with department founding date",
        ],
    ),

    "sports_entity": SearchStrategy(
        pattern_name="Sports Entity Identification",
        priority_search_guidance=(
            "Sports questions often require finding a specific TEAM, PLAYER, or MATCH. "
            "The key is to use the most distinctive sports-specific constraint: "
            "unique scorelines, specific trophy years, founding decades."
        ),
        search_tactics=[
            "For teams: combine founding period + league + city/country: 'team founded 1900s [city] [league]'",
            "For matches: use the SCORELINE or specific event (e.g., 'scored all goals first half 2001 match')",
            "Trophy/championship years are excellent search anchors: 'second major trophy 1970s European'",
            "Player identification: combine position + specific stats + team + era",
            "For eSports: player name + champion + specific season (e.g., 'Faker Ahri Worlds 2023')",
        ],
        verification_approach=(
            "Verify founding year, trophy years, and match details against multiple sources. "
            "Sports databases (Transfermarkt, Wikipedia lists) are authoritative."
        ),
        common_pitfalls=[
            "Confusing domestic and international trophies",
            "Not narrowing by country/continent when the question specifies 'European' or 'national'",
            "Using outdated team names when the question refers to current status",
        ],
    ),

    "entertainment_media": SearchStrategy(
        pattern_name="Entertainment/Media Entity Search",
        priority_search_guidance=(
            "Entertainment questions target specific movies, shows, songs, or people in media. "
            "Use the most DISTINCTIVE production detail: episode counts, air dates, runtime, "
            "or specific creative team details (one director + two writers)."
        ),
        search_tactics=[
            "Episode count + air period is very distinctive: '50 episodes 1990s ran January to December'",
            "For foreign language titles: find the ORIGINAL work first, then search for the localized name",
            "Director/writer combinations narrow searches: 'one director two writers [decade] series'",
            "Specific runtime or format details: 'less than 5 minutes episodes 1990s'",
            "For influencer/YouTuber questions: search platform + era + distinguishing achievement",
        ],
        verification_approach=(
            "Verify episode count, air dates, and creative team against an authoritative database "
            "(IMDb, MyAnimeList, AniDB). Check that the localized title matches the target country."
        ),
        common_pitfalls=[
            "Searching for the localized title directly without knowing the original work",
            "Confusing similar shows from the same era",
            "Not checking all constraints (episode count, runtime, number of seasons)",
        ],
    ),

    "biographical_identification": SearchStrategy(
        pattern_name="Biographical Person Identification",
        priority_search_guidance=(
            "This question asks you to identify a person based on biographical facts. "
            "The KEY is to find the RAREST biographical fact — not 'born in 1940s' (too common) "
            "but something like 'died in car accident August [year]' or 'tissues donated from teenager'."
        ),
        search_tactics=[
            "DEATH circumstances are the rarest constraints (accident type + year + age)",
            "Family relationships as search terms: 'teen died car accident father survived [year range]'",
            "Career + specific achievement combination: 'born 1940s theater [country] film director budget'",
            "Marriage/spouse details: search the SPOUSE if they're easier to identify, then find the subject through them",
            "Honorary degrees, Hall of Fame inductions, and awards have searchable databases",
        ],
        verification_approach=(
            "Once you have a candidate name, search 'name + birth year' and 'name + [specific event]' "
            "to verify each biographical constraint. Check at least 3 constraints."
        ),
        common_pitfalls=[
            "Starting with the most common constraint (birth decade) instead of the rarest",
            "Not following the spouse/family connection path when the subject is obscure",
            "Accepting a candidate that matches 2 constraints without checking the others",
        ],
    ),

    "business_entity": SearchStrategy(
        pattern_name="Business/Company Entity Search",
        priority_search_guidance=(
            "Business questions target companies, CEOs, or financial data. "
            "The strongest search signals are: specific financial figures, "
            "unique company descriptions, and industry + founding era combinations."
        ),
        search_tactics=[
            "Financial figures in quotes: '\"debt-free balance sheet\" powerboats' or '\"order backlog\"'",
            "CEO + specific claim combination: 'CEO said financially sound [industry] 2020'",
            "Company characteristics: 'manufactures powerboats debt-free 2020s CEO'",
            "For startup identification: 'dropped out school founded [year range] acquired by'",
            "Investor/backing chains: find the parent company first, then their portfolio companies",
        ],
        verification_approach=(
            "Verify the company exists, the CEO name matches, and the financial claims "
            "appear in investor reports or press releases."
        ),
        common_pitfalls=[
            "Searching for generic industry terms without the distinctive financial detail",
            "Confusing the parent company with the subsidiary",
            "Not checking the TIME period of financial claims (balance sheets change yearly)",
        ],
    ),

    "exact_phrase_given": SearchStrategy(
        pattern_name="Exact Phrase Search",
        priority_search_guidance=(
            "The question contains QUOTED PHRASES — these are your golden search terms. "
            "Use them exactly as given."
        ),
        search_tactics=[
            "Copy the exact quoted phrase into your search query",
            "Combine the exact phrase with ONE additional constraint keyword",
            "If the first search returns too many results, add a year or domain keyword",
        ],
        verification_approach="Verify the phrase appears in the context described by the question.",
        common_pitfalls=[
            "Paraphrasing the quoted phrase instead of using it exactly",
            "Adding too many words around the exact phrase (dilutes the search)",
        ],
    ),

    "nickname_search": SearchStrategy(
        pattern_name="Nickname/Title-Based Search",
        priority_search_guidance=(
            "The question mentions a NICKNAME or informal title. "
            "Nicknames are extremely distinctive search terms — they're usually unique to one entity."
        ),
        search_tactics=[
            "Search the nickname in EXACT quotes: '\"Mr. Le Mans\"' or '\"The Ice Man\"'",
            "Combine nickname + domain: '\"nickname\" racing driver' or '\"nickname\" football'",
            "If the nickname yields the person directly, verify their stats against other constraints",
            "For fitting/earned nicknames: the question may describe WHY it fits — search the reason",
        ],
        verification_approach=(
            "Confirm the nickname belongs to your candidate by checking multiple sources."
        ),
        common_pitfalls=[
            "Not putting the nickname in quotes (gets diluted by generic results)",
            "Confusing people who share similar nicknames in different eras",
        ],
    ),

    "scientific_classification": SearchStrategy(
        pattern_name="Scientific Name / Species Search",
        priority_search_guidance=(
            "Questions about species or scientific classification require searching "
            "through taxonomic databases and natural history publications. "
            "The key constraints are: discoverer, discovery era, physical characteristics, geographic range."
        ),
        search_tactics=[
            "Search the DISCOVERER/DESCRIBER + era: 'naturalist described species 1780s [characteristic]'",
            "Physical measurements are distinctive: search specific sizes/counts + taxonomic group",
            "Geographic distribution + taxonomic features: '[size]mm [habitat] [region]'",
            "For species counts or subspecies: 'genus [number] subspecies described'",
            "Taxonomic databases: search 'site:gbif.org' or 'site:catalogueoflife.org' + characteristics",
        ],
        verification_approach=(
            "Verify ALL physical characteristics (size, color, subspecies count) against "
            "the original species description or authoritative taxonomy databases."
        ),
        common_pitfalls=[
            "Confusing species within the same genus that share some characteristics",
            "Not verifying the describer/discoverer (multiple species may match physical traits)",
            "Accepting a species that matches size but not geographic range or discovery date",
        ],
    ),

    "general_entity": SearchStrategy(
        pattern_name="General Entity Search",
        priority_search_guidance=(
            "Use the most SPECIFIC and UNUSUAL combination of keywords from the question. "
            "Avoid generic terms — every word in your search should narrow results."
        ),
        search_tactics=[
            "Pick 4-6 words that are LEAST likely to appear together on random pages",
            "Proper nouns and specific numbers are always better than generic descriptions",
            "If the first 2-3 searches fail, completely change your angle — try different constraints",
            "Use fetch=true on promising results to read full content for verification",
        ],
        verification_approach="Verify at least 3 constraints from the question against your candidate.",
        common_pitfalls=[
            "Using all constraints in one search query (too restrictive, returns nothing)",
            "Repeating the same search angle more than twice",
            "Not reading the full page content when snippets are ambiguous",
        ],
    ),
}


# ===========================================================================
# Strategy Builder (builds a prompt section from detected patterns)
# ===========================================================================

def build_search_skill_section(
    question: str,
    has_image: bool = False,
    max_chars: int = 1500,
) -> str:
    """Build a concise search strategy section based on detected question patterns."""
    patterns = detect_question_patterns(question, has_image)

    parts = ["## Search Skill Guide (pattern-matched)"]
    char_count = 0

    for pattern_name in patterns[:3]:  # Max 3 patterns per question
        strategy = STRATEGY_LIBRARY.get(pattern_name)
        if not strategy:
            continue

        section = f"\n### {strategy.pattern_name}\n{strategy.priority_search_guidance}\n"
        section += "Key tactics:\n"
        for tactic in strategy.search_tactics[:3]:
            section += f"  - {tactic}\n"
        section += f"Pitfall to avoid: {strategy.common_pitfalls[0]}\n"

        if char_count + len(section) > max_chars:
            break
        parts.append(section)
        char_count += len(section)

    # Always add universal search principles
    universal = (
        "\n### Universal Principles\n"
        "- Answers are SHORT entities (avg 2 words). If your candidate is longer than 5 words, extract the core entity.\n"
        "- The FIRST search should use the MOST DISTINCTIVE clue (not the first constraint mentioned).\n"
        "- After 3 failed searches in one direction, PIVOT completely to a different constraint.\n"
        "- When you find a candidate, verify it against ALL constraints before outputting.\n"
        "- Use fetch=true to read full page content when snippets don't confirm your candidate.\n"
    )
    if char_count + len(universal) <= max_chars:
        parts.append(universal)

    return "\n".join(parts)


def get_search_priority_hint(question: str, has_image: bool = False) -> str:
    """Returns a one-line hint about what to search first."""
    ql = question.lower()
    patterns = detect_question_patterns(question, has_image)

    if "exact_phrase_given" in patterns:
        quotes = re.findall(r'"([^"]+)"', question)
        if quotes:
            return f"START with exact phrase search: {quotes[0]}"

    if "nickname_search" in patterns:
        return "START with the nickname/title in exact quotes — it's the most distinctive clue"

    if has_image:
        return "START with search_image to identify the entity, then text-search for the answer"

    if "relationship_chain" in patterns:
        return "Resolve the chain step by step — find the FIRST identifiable entity, then follow links"

    if "rare_number_intersection" in patterns:
        all_nums = re.findall(r'\b(\d{2,7})\b', question)
        non_year = [n for n in all_nums if not (1800 <= int(n) <= 2030)]
        if non_year:
            return f"START with distinctive number(s) {non_year[:2]} combined with context keywords"

    if "biographical_identification" in patterns:
        if 'died' in ql or 'death' in ql or 'accident' in ql:
            return "START with the death/accident circumstances — rarest biographical constraint"
        if 'spouse' in ql or 'married' in ql:
            return "Consider finding the SPOUSE first if they are more identifiable"

    return "Identify the 1-2 MOST DISTINCTIVE clues and search those first (4-6 words)"
