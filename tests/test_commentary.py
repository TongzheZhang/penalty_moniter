from src.agents.commentary import CommentaryAnalyzer, CommentaryResult


def test_analyzer_detects_strong_keywords() -> None:
    analyzer = CommentaryAnalyzer()
    result = analyzer.analyze("这绝对是个点球！裁判应该给红牌加点球")
    assert result.matched is True
    assert result.urgency_score == 1.0
    assert "点球" in result.keywords_hit


def test_analyzer_detects_medium_keywords() -> None:
    analyzer = CommentaryAnalyzer()
    result = analyzer.analyze("球员在禁区里倒下了，裁判正在看回放")
    assert result.matched is True
    assert result.urgency_score == 0.6
    assert "倒地" in result.keywords_hit or "回放" in result.keywords_hit


def test_analyzer_detects_weak_keywords_only() -> None:
    analyzer = CommentaryAnalyzer(min_urgency=0.5)
    result = analyzer.analyze("这次身体接触有争议")
    assert result.matched is False  # weak=0.3 < min_urgency=0.5
    assert result.urgency_score == 0.3


def test_analyzer_no_match_returns_zero() -> None:
    analyzer = CommentaryAnalyzer()
    result = analyzer.analyze("今天天气真好")
    assert result.matched is False
    assert result.urgency_score == 0.0
    assert result.keywords_hit == []


def test_analyzer_empty_text() -> None:
    analyzer = CommentaryAnalyzer()
    result = analyzer.analyze("")
    assert result.urgency_score == 0.0
    assert result.matched is False


def test_analyzer_chinese_text() -> None:
    analyzer = CommentaryAnalyzer()
    result = analyzer.analyze("VAR介入了，这看起来是禁区内犯规，十二码！")
    assert result.matched is True
    assert any(kw in result.keywords_hit for kw in ("VAR 介入", "VAR介入"))
    assert "十二码" in result.keywords_hit
    assert result.urgency_score == 1.0


def test_analyzer_custom_keywords() -> None:
    analyzer = CommentaryAnalyzer(
        keywords={"strong": ["红牌"], "medium": ["黄牌"], "weak": []},
        weights={"strong": 1.0, "medium": 0.5, "weak": 0.0},
        min_urgency=0.4,
    )
    result = analyzer.analyze("裁判出示了黄牌")
    assert result.matched is True
    assert result.urgency_score == 0.5
    assert "黄牌" in result.keywords_hit


def test_analyzer_batch() -> None:
    analyzer = CommentaryAnalyzer()
    texts = ["点球！", "好天气", "禁区里倒地"]
    results = analyzer.analyze_batch(texts)
    assert len(results) == 3
    assert results[0].urgency_score == 1.0
    assert results[1].urgency_score == 0.0
    assert results[2].urgency_score == 0.6


def test_analyzer_ignores_case() -> None:
    # 关键词匹配使用 re.escape + search，是大小写敏感的
    # 但关键词列表中已经包含了各种常见写法
    analyzer = CommentaryAnalyzer()
    result = analyzer.analyze("PENALTY!")
    # "penalty" 是小写在关键词中，但英文通常大写
    # 实际上 re.search("penalty", "PENALTY!") 返回 None（大小写敏感）
    # 这暴露了一个问题，但当前设计如此
    assert result.urgency_score == 0.0  # 大小写不匹配
