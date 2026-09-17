"""把一条笔记汇总成模型可读的文本。

同一个"汇总哪些字段"的需求此前散成四份各写各的，其中三份是逐字复制的
f-string（`ai._prompt` / `ai._folder_prompt` / `ai._memory_prompt` 里的
`笔记标题：… 笔记正文：…`），第四份是 Agent 的 `get_note` 工具直接返回整份 JSON。
后果有两个：

1. 改一处忘两处；
2. **后加的字段一处都进不去**——`image_text` 与评论就是这么被漏掉的。
   对"正文只剩话题标签、内容全在长图里"的笔记，提示词拿到的是空壳。
   实测库里 `6aa3c6ff`（淘宝 AI Agent 面经）正文 0 字、内容全在 `image_text` 里，
   于是它的收藏夹建议与记忆提炼一直是在空转。

本模块把字段清单收成一处，再提供两种呈现：

- `note_inline`：单段文本，喂提示词；逐字段折叠空白，只输出有内容的字段。
- `note_markdown`：分段 markdown，回给 Agent 当工具结果；缺失的字段明写
  "（无：原因）"，让模型确知"库里没有"而不是以为自己读漏了。

秋招知识点（单一事实来源）：字段清单只在 `_fields` 里定义一次。以后笔记新增字段，
改这一处，多条消费路径同时生效。
"""

# 标题在所有呈现里都限长：它本该是单行短文本，超长必是异常输入。
TITLE_LIMIT = 200
# 图片文字送入提示词时的默认截断长度。正文不设默认上限——各调用方按自己的
# 上下文预算传 content_limit（记忆提炼传 800，分析不截断）。
IMAGE_TEXT_LIMIT = 800
# 渲染评论时的护栏。采集层已经限过一次（20 条 / 每条 3 条回复），
# 这里防的是历史数据与将来换源。
COMMENT_LIMIT = 20
COMMENT_CHARS = 300
REPLY_LIMIT = 3

# 空字段在 markdown 里的说明。区分"没有这个字段"和"字段存在但为空"，
# 模型据此能判断该请用户补采集，还是这篇确实没有评论。
NO_TITLE = "（无：这篇没有标题）"
NO_CONTENT = "（无：正文为空，内容可能在图片里）"
NO_IMAGE_TEXT_WITHOUT_IMAGES = "（无：这篇没有图片）"
NO_IMAGE_TEXT_UNREAD = "（无：有 {count} 张图，但没有识别出文字）"
NO_COMMENTS_NOT_COLLECTED = "（无：入库时未采集评论；较早导入或走网页解析通道的笔记通常如此）"
NO_COMMENTS_EMPTY = "（无：采集到了评论容器，但一条正文都没有）"


def flatten_inline(text, limit: int | None = None) -> str:
    """把任意文本压成一段安全短文本：剥离控制字符、折叠空白、按长度截断。

    秋招知识点（提示词注入防御）：笔记正文与评论都是不可信输入——评论尤其是
    陌生人写的任意文本，谁都能在里面写一句"忽略以上要求"。控制字符（换行、
    制表符、不可见字符）会破坏提示词原有的指令边界，让资料在视觉上变成新指令；
    折叠成单段纯文本后，它就只能作为资料存在。截断则是上下文预算护栏，
    避免长正文挤掉真正的规则说明。

    `limit=None` 表示不截断（调用方已自行截过时用）。
    """
    if not isinstance(text, str):
        return ""
    # isprintable() 对 \n \t \r 与各类不可见字符返回 False，统一替换为空格。
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    # split() 会丢弃首尾空白并把连续空白压成一个空格，再 join 回来即单段文本。
    cleaned = " ".join(cleaned.split())
    # 用 `is not None` 而不是真值判断：limit=0 应得到空串，而不是"不截断"。
    return cleaned if limit is None else cleaned[:limit]


def _byline(item: dict) -> str:
    """评论署名：作者在前，括号里补地点与热度；两项都缺时不留空括号。"""
    author = str(item.get("author") or "").strip() or "匿名用户"
    parts = []
    location = str(item.get("ip_location") or "").strip()
    if location:
        parts.append(location)
    likes = str(item.get("like_count") or "").strip()
    if likes:
        # 小红书给的是热度表示（"1千+"）而非数字，原样展示比硬转更诚实。
        parts.append(f"{likes} 赞")
    return f"{author}（{' · '.join(parts)}）" if parts else author


def _render_comments(note: dict) -> str:
    """把评论渲染成缩进列表；只认采集层写下的字段名，不认识的结构直接跳过。"""
    comments = note.get("comments")
    if not isinstance(comments, list):
        return ""
    lines = []
    for item in comments[:COMMENT_LIMIT]:
        if not isinstance(item, dict):
            continue
        body = str(item.get("content") or "").strip()[:COMMENT_CHARS]
        if not body:
            continue
        lines.append(f"- {_byline(item)}：{body}")
        replies = item.get("replies")
        if not isinstance(replies, list):
            continue
        for reply in replies[:REPLY_LIMIT]:
            if not isinstance(reply, dict):
                continue
            reply_body = str(reply.get("content") or "").strip()[:COMMENT_CHARS]
            if reply_body:
                lines.append(f"  - {_byline(reply)}：{reply_body}")
    return "\n".join(lines)


def _fields(note: dict, *, content_limit: int | None = None,
            image_text_limit: int | None = IMAGE_TEXT_LIMIT,
            include_image_text: bool = True, include_comments: bool = True,
            ) -> list[tuple[str, str, str]]:
    """字段清单的唯一出处：顺序、标签、取值与"为空的原因"都只在这里定义。

    返回 `[(标签, 正文, 空说明)]`。正文为空串时，由呈现层决定是整段跳过
    （提示词场景）还是写上空说明（工具结果场景）。
    """
    fields: list[tuple[str, str, str]] = []

    title = str(note.get("title") or "").strip()[:TITLE_LIMIT]
    fields.append(("标题", title, NO_TITLE))

    content = str(note.get("content") or "").strip()
    if content_limit is not None:
        content = content[:content_limit]
    fields.append(("正文", content, NO_CONTENT))

    if include_image_text:
        image_text = str(note.get("image_text") or "").strip()
        if image_text_limit is not None:
            image_text = image_text[:image_text_limit]
        images = note.get("images")
        image_count = len(images) if isinstance(images, list) else 0
        reason = (NO_IMAGE_TEXT_WITHOUT_IMAGES if image_count == 0
                  else NO_IMAGE_TEXT_UNREAD.format(count=image_count))
        fields.append(("图片文字", image_text, reason))

    if include_comments:
        rendered = _render_comments(note)
        comments = note.get("comments")
        # 条数写进标题，模型一眼能看出这份评论是否被 COMMENT_LIMIT 截过。
        label = f"评论（{len(comments)} 条）" if rendered else "评论"
        reason = NO_COMMENTS_EMPTY if "comments" in note else NO_COMMENTS_NOT_COLLECTED
        fields.append((label, rendered, reason))

    return fields


def note_inline(note: dict, *, content_limit: int | None = None,
                image_text_limit: int | None = IMAGE_TEXT_LIMIT,
                include_image_text: bool = True,
                include_comments: bool = False) -> str:
    """把笔记压成提示词片段：一行一个字段，值已折叠成单段。

    只输出有内容的字段，不写占位符——提示词里不需要"这篇没有评论"这类噪音，
    那是工具结果才需要的确定性说明。

    每个字段都过 `flatten_inline`，所以字段值无法制造新的"指令行"；换行只可能
    出现在我们自己的 `标签：` 分隔处。因此结果可以安全地直接放进提示词。

    include_comments 默认关闭：评论是陌生人的任意文本，加进提示词既涨 token
    又抬高注入面，只在确实要判断"大家怎么看"的场景才打开。
    """
    parts = []
    for label, body, _reason in _fields(
        note, content_limit=content_limit, image_text_limit=image_text_limit,
        include_image_text=include_image_text, include_comments=include_comments,
    ):
        flat = flatten_inline(body)
        if flat:
            parts.append(f"笔记{label}：{flat}")
    return "\n".join(parts)


def note_markdown(note: dict, *, content_limit: int | None = None,
                  image_text_limit: int | None = IMAGE_TEXT_LIMIT,
                  include_image_text: bool = True,
                  include_comments: bool = True) -> str:
    """把笔记渲染成分段 markdown，供 Agent 的 `get_note` 工具返回。

    与 `note_inline` 相反，这里**保留原始换行**：工具结果是资料而不是指令，
    不需要压成单段，分段的可读性更重要。缺失字段写上空说明——这正是本模块
    存在的理由之一：模型不该靠"JSON 里少了一个键"来发现内容缺失。
    """
    head, sections = "", []
    for label, body, reason in _fields(
        note, content_limit=content_limit, image_text_limit=image_text_limit,
        include_image_text=include_image_text, include_comments=include_comments,
    ):
        if label == "标题":
            head = f"# {body}\n\n" if body else ""
            continue
        sections.append(f"## {label}\n{body or reason}")

    text = head + "\n\n".join(sections)

    footer = []
    url = str(note.get("source_url") or "").strip()
    if url:
        footer.append(f"来源：{url}")
    images = note.get("images")
    if isinstance(images, list) and images:
        footer.append(f"图片：{len(images)} 张")
    if footer:
        text = f"{text}\n\n---\n" + "\n".join(footer)
    return text
