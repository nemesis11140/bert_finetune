"""数据：模板合成训练数据 + Dataset 封装。

上半部分是模板和词库（想加新句式/词汇直接在这里扩充），
下半部分是训练用的 Dataset。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

# ==================== 模板库 ====================
# 正面句子模板
positive_templates = [
    "这部电影真的很{adj}，我{adv}喜欢！",
    "今天天气{adj}，心情{adv}好。",
    "这家餐厅的{food}非常{adj}，推荐大家来！",
    "我终于{action}了，太{adj}了！",
    "这个{product}质量{adj}，{adv}满意。",
    "老师讲得{adj}，我{adv}听懂了很多。",
    "这次旅行{adj}，风景{adv}美。",
    "男朋友送我的{gift}太{adj}了，好开心！",
    "这本书内容{adj}，{adv}值得一读。",
    "公司氛围{adj}，同事们都{adv}友好。",
    "这个{app}用起来{adv}方便，界面{adj}。",
    "孩子今天考试{adj}，考了100分！",
    "这个{color}的衣服{adv}适合我。",
    "演唱会{adv}精彩，歌手的嗓音{adj}。",
    "这个{plan}确实{adv}合理，大家一致通过。",
    "今天食堂的饭菜{adj}，我吃了两碗。",
    "新买的{device}性能{adj}，运行{adv}流畅。",
    "女朋友做的{meal}太{adj}了，好幸福！",
    "这个{game}游戏{adv}好玩，根本停不下来。",
    "健身房环境{adj}，器械{adv}齐全。",
    "闺蜜推荐的{product}果然{adj}，用起来{adv}顺手。",
    "这家店的{service}真{adj}，店员{adv}热情。",
    "周末和朋友们去{place}玩，{adv}开心。",
    "这个{game}的画面{adj}，配乐也{adv}好听。",
    "妈妈做的饭菜永远最{adj}，吃一口就{adv}幸福。",
    "这个{app}更新后{adv}好用，新增的功能{adj}。",
    "收到offer那一刻{adv}激动，努力总算没有白费。",
    "这个{place}的环境{adj}，让人{adv}放松。",
    "这个{product}性价比{adj}，{adv}值得购买。",
    "运动完出一身汗，感觉{adv}畅快！",
    "朋友送的花{adj}，闻着{adv}香。",
    "这次{event}过得{adv}开心，留下了{adj}回忆。",
]

# 正面情绪中，用否定词反衬正面（"不差"→好）的模板
positive_negation_templates = [
    "这个{product}一点都不{neg_adj}，用起来{adv}顺手。",
    "这家餐厅的{service}一点也不{neg_adj}，{adv}满意。",
    "这本书一点都不{neg_adj}，{adv}吸引人。",
    "这次旅行并不{neg_adj}，体验{adv}棒。",
    "这个{app}并不卡顿，用起来{adv}流畅。",
    "这个{device}一点也不{neg_adj}，性价比{adv}高。",
    "这个{game}并不无聊，{adv}好玩。",
]

# 负面句子模板
negative_templates = [
    "这部电影太{adj}了，我{adv}看不下去。",
    "今天下雨，心情{adv}糟糕。",
    "这家餐厅的{service}太{adj}了，不会再来了。",
    "我的{device}又{action}了，真是{adj}。",
    "这个{product}质量{adj}，{adv}失望。",
    "老师讲得{adj}，我完全听不懂。",
    "这次旅行{adj}，体验{adv}差。",
    "男朋友忘记了我的{event}，好{adj}。",
    "这本书内容{adj}，{adv}看不进去。",
    "公司加班太{adj}了，领导{adv}压榨。",
    "这个{app}经常{action}，体验{adv}差。",
    "孩子今天考试{adj}，被老师批评了。",
    "这个{color}的衣服{adv}不适合我。",
    "演唱会{adv}让人失望，音效{adj}。",
    "这个{plan}方案{adv}不合理，大家都不满意。",
    "今天食堂的饭菜{adj}，我只吃了一口。",
    "新买的{device}经常{action}，太{adj}了。",
    "女朋友心情{adj}，我也不知道怎么安慰。",
    "这个{game}游戏{adv}无聊，浪费时间。",
    "健身房人太{adj}了，器械都{action}。",
    "这个{product}用了两天就{action}，{adv}后悔。",
    "早高峰的地铁{adv}拥挤，体验{adj}。",
    "这家店的{service}太{adj}了，态度{adv}差。",
    "这本书写得{adj}，读了半小时就{action}了。",
    "周末还加班到深夜，感觉{adj}透了。",
    "这个{game}的匹配机制{adv}差，还经常{action}。",
    "这个{app}老是{action}，用起来{adv}烦。",
    "这家{place}的环境{adj}，我再也不想去了。",
    "这个{plan}推进{adv}困难，结果{adj}。",
    "外卖送了一个小时，饭菜都{adj}了。",
    "这次{event}办得{adj}，大家都不满意。",
    "新剪的发型{adj}，看着{adv}别扭。",
    "最近工作压力{adv}大，整个人{adj}。",
]

# 负面情绪中，用否定词反衬负面（"没那么好"→差）的模板
negative_negation_templates = [
    "这个{product}并没有想象中{pos_adj}，{adv}失望。",
    "这部电影并不{pos_adj}，剧情{adv}平淡。",
    "这家餐厅并没有那么{pos_adj}，体验{adv}一般。",
    "这个{app}并没有宣传的{pos_adj}，{adv}难用。",
    "这次的{event}并不{pos_adj}，{adv}无聊。",
]

# ==================== 转折句式模板 ====================
# 上面几组模板有个共同毛病：一句话里所有分句极性一致，模型只要看到褒义词就判正面。
# 「这电影真好看，看得我昏昏欲睡」这类句子正是被这个捷径坑了。
# 下面三组专门制造前后分句极性相反的句子，让模型学会看转折连词和末句分句定调。

# 先抑后扬：前半句挑刺，后半句落地在正面 → label 1
positive_contrast_templates = [
    "这家餐厅的{service}一般，但{food}是真的好吃，{emph}满意。",
    "这个{app}的界面不算好看，功能却很齐全，用着{emph}顺手。",
    "虽然过程{emph}曲折，好在{action}了，{emph}开心。",
    "前几集有点拖沓，越看越上头，{emph}推荐。",
    "这个{product}价格不便宜，不过用起来{emph}顺手，值了。",
    "开头有点平淡，看到后面{emph}精彩，{emph}喜欢。",
    "虽然{device}有点小毛病，但{emph}好用，{emph}划算。",
    "这次{event}虽然出了点小插曲，但整体{emph}开心。",
    "上班是累了点，不过同事都{emph}友好，氛围也轻松。",
    "这本书开头有点平淡，读下去才发现内容{emph}深刻。",
    "这个{place}看着一般，进去才发现{emph}舒服，{emph}放松。",
]

# 先扬后抑：前半句夸得响亮，后半句才是真实体验 → label 0
negative_contrast_templates = [
    "这部电影{emph}好看，看得我{bad_result}。",
    "开头还挺{praise}的，后面越看越{complain}。",
    "这个{product}外观{praise}，用起来却{emph}卡顿。",
    "宣传得{emph}好，到手才发现质量{complain}。",
    "看着挺{praise}，一上手就{action}，{emph}糟心。",
    "刚开始{emph}满意，用了两天就{action}了。",
    "前几集确实{praise}，可惜后面越来越{complain}。",
    "表面看着{praise}，实际体验{emph}差。",
    "演员阵容{emph}强，可惜剧情{complain}，{emph}失望。",
    "预约的时候{emph}热情，到店就没人理了。",
    "包装{praise}，拆开一看东西{emph}粗糙。",
]

# 反讽：嘴上全是好话，落点却是明确的负面 → label 0
negative_sarcasm_templates = [
    "呵呵，这服务真是{praise_service}，好到让我想投诉。",
    "真是{praise_service}啊，等了两个小时才上菜。",
    "这家店的服务实在{praise_service}，我这辈子都不想再来第二次。",
    "都说这{product}{praise}，买回来三天就{action}。",
    "厉害厉害，这服务真是{praise_service}，厉害到我再也不敢来了。",
    "感谢商家的{praise_service}服务，让我体验了一把什么叫花钱买罪受。",
    "这{product}真有{praise_noun}，用了两天就{action}，我谢谢你。",
]

# ==================== 词库 ====================
POSITIVE_ADJ = [
    "精彩", "棒", "优秀", "出色", "完美", "好", "美味", "漂亮", "舒服", "温馨",
    "精致", "高级", "专业", "耐心", "周到", "干净", "热闹", "美丽", "壮观", "神奇",
    "满意", "贴心", "诱人", "实惠", "靠谱", "惊艳", "顺手", "亮眼", "舒适", "划算",
]
NEGATIVE_ADJ = [
    "糟糕", "差", "垃圾", "烂", "失望", "难吃", "无聊", "吵闹", "老旧", "粗糙",
    "一般", "冷淡", "不耐烦", "脏乱", "拥挤", "吓人", "难看", "枯燥", "乏味", "糟心",
    "闹心", "坑人", "气人", "离谱", "破", "差劲", "憋屈", "烦人", "刺耳", "油腻",
]
POSITIVE_ADV = [
    "非常", "特别", "十分", "超级", "很", "太", "真的", "相当", "极其", "尤其",
    "确实", "简直", "真心", "格外", "无比", "完全",
]
NEGATIVE_ADV = [
    "非常", "特别", "十分", "超级", "很", "太", "真的", "相当", "极其", "完全",
    "简直", "实在", "真心",
]

# 转折句后半段的"真实体验差"短语，配 negative_contrast_templates 用
BAD_RESULT = [
    "昏昏欲睡", "直打瞌睡", "坐立难安", "看不下去", "想快进", "玩了一路手机", "睡了一觉",
]

# 转折句式专用小词库。这些位置只吃"能修饰事物/体验/服务"的词，
# 直接复用 POSITIVE_ADJ / NEGATIVE_ADJ 全量词库会生成
# "氛围实惠""这背包真有壮观""质量无聊" 这类搭配不通的句子，等于往训练集里灌噪声。
PRAISE_ADJ = ["不错", "很棒", "出色", "惊艳", "靠谱", "精致", "高级", "亮眼", "出彩"]
COMPLAIN_ADJ = ["一般", "普通", "无聊", "粗糙", "拉胯", "敷衍", "差劲"]
PRAISE_SERVICE = ["贴心", "周到", "专业", "热情", "耐心", "高效", "细致"]
PRAISE_NOUN = ["档次", "水平", "品质", "排面", "质感"]
# 通用强调副词：不含"太"——"太"后面必须跟"了"，放动词前会生成"太推荐"这种病句
EMPH_ADV = ["非常", "特别", "超级", "十分", "相当", "真的", "格外", "真心"]

FOOD = ["红烧肉", "水煮鱼", "糖醋排骨", "麻婆豆腐", "宫保鸡丁", "炒时蔬", "汤", "甜点", "烤鸭", "火锅",
        "酸菜鱼", "小龙虾", "烧烤", "生煎", "牛肉面", "卤味", "蒸鱼", "虾饺"]
# 注意：这里的词不要再带"了"后缀——有的模板写成 "{action}了"，带上会生成
# "我终于上岸了了" 这种病句；不带后缀的模板（"经常{action}"）同样读得通。
ACTION_POS = ["成功", "完成", "实现", "达成", "做到", "实现目标", "通过", "上岸", "拿下", "搞定"]
ACTION_NEG = ["坏掉", "卡顿", "崩溃", "死机", "闪退", "关机", "没电", "出问题", "黑屏", "断网",
              "没信号", "掉线", "失灵", "卡死", "延迟"]
PRODUCT = ["手机", "电脑", "耳机", "键盘", "鼠标", "显示器", "音响", "相机", "手表", "背包",
           "平板", "智能手表", "扫地机器人", "投影仪", "路由器", "摄像头"]
GIFT = ["项链", "手表", "香水", "鲜花", "蛋糕", "玩具", "书", "画", "音乐盒", "手链",
        "围巾", "口红", "巧克力", "玩偶"]
APP = ["微信", "支付宝", "美团", "抖音", "小红书", "网易云音乐", "淘宝", "京东", "B站", "知乎",
       "拼多多", "腾讯视频", "爱奇艺", "keep"]
COLOR = ["红色", "蓝色", "白色", "黑色", "粉色", "紫色", "金色", "银色", "绿色", "黄色", "杏色", "深蓝", "米白"]
DEVICE = ["手机", "电脑", "平板", "耳机", "充电器", "笔记本", "台式机", "音响", "路由器", "摄像头",
          "扫地机器人", "智能音箱", "显示器"]
MEAL = ["早饭", "午饭", "晚饭", "面条", "饺子", "披萨", "汉堡", "寿司", "烤肉", "火锅",
        "炒饭", "煎饼", "拉面", "咖喱", "意面"]
GAME = ["王者荣耀", "原神", "和平精英", "英雄联盟", "魔兽世界", "梦幻西游", "阴阳师", "明日方舟",
        "崩坏3", "第五人格", "星穹铁道", "金铲铲", "蛋仔派对", "三角洲行动"]
SERVICE = ["服务", "卫生", "环境", "态度", "上菜速度", "价格", "口味", "装修", "地段", "停车"]
EVENT = ["生日", "纪念日", "情人节", "母亲节", "父亲节", "元旦", "春节", "中秋节", "圣诞节", "七夕",
         "结婚纪念日", "生日聚会", "毕业典礼"]
PLAN = ["计划", "方案", "提议", "想法", "设计", "布局", "策划", "项目", "点子"]
PLACE = ["公园", "海边", "古镇", "电影院", "游乐园", "商场", "图书馆", "咖啡馆", "健身房", "山上",
         "西湖", "夜市", "博物馆", "滑雪场", "果园"]

positive_words = {
    "adj": POSITIVE_ADJ,
    "adv": POSITIVE_ADV,
    "pos_adj": POSITIVE_ADJ,     # 正面对话中强调"好"时用
    "neg_adj": NEGATIVE_ADJ,     # 正面对话中否定"差"时用
    "emph": EMPH_ADV,            # 转折句式专用，见上方小词库注释
    "praise": PRAISE_ADJ,
    "complain": COMPLAIN_ADJ,
    "praise_service": PRAISE_SERVICE,
    "praise_noun": PRAISE_NOUN,
    "food": FOOD,
    "action": ACTION_POS,
    "product": PRODUCT,
    "gift": GIFT,
    "app": APP,
    "color": COLOR,
    "device": DEVICE,
    "meal": MEAL,
    "game": GAME,
    "service": SERVICE,
    "event": EVENT,
    "plan": PLAN,
    "place": PLACE,
}

negative_words = {
    "adj": NEGATIVE_ADJ,
    "adv": NEGATIVE_ADV,
    "pos_adj": POSITIVE_ADJ,     # 负面情绪中否定"好"时用
    "neg_adj": NEGATIVE_ADJ,
    "bad_result": BAD_RESULT,    # 转折句后半段的"实际体验"落点
    "emph": EMPH_ADV,            # 转折句式专用，见上方小词库注释
    "praise": PRAISE_ADJ,
    "complain": COMPLAIN_ADJ,
    "praise_service": PRAISE_SERVICE,
    "praise_noun": PRAISE_NOUN,
    "service": SERVICE,
    "device": DEVICE,
    "product": PRODUCT,
    "app": APP,
    "action": ACTION_NEG,
    "color": COLOR,
    "event": EVENT,
    "plan": PLAN,
    "game": GAME,
    "meal": MEAL,
    "gift": GIFT,
    "place": PLACE,
}


def _generate_text(templates, word_pool, label, num_samples, seen):
    """按模板生成 num_samples 条不重复的训练数据"""
    data = []
    tries = 0
    max_tries = num_samples * 50  # 防死循环的兜底
    while len(data) < num_samples and tries < max_tries:
        tries += 1
        template = random.choice(templates)
        text = template
        for key, values in word_pool.items():
            if f"{{{key}}}" in text:
                text = text.replace(f"{{{key}}}", random.choice(values))
        if text in seen:  # 去重
            continue
        seen.add(text)
        data.append({"text": text, "label": label})
    if len(data) < num_samples:
        print(f"  ⚠️ 模板组合已用尽，实际生成 {len(data)} 条（目标 {num_samples} 条）")
    return data


def generate(processed_path: str | Path, num_per_class: int, seed: int) -> dict:
    """生成正/负样本并写入 processed_path，返回统计信息。"""
    random.seed(seed)

    seen: set[str] = set()
    positive = _generate_text(
        positive_templates + positive_negation_templates + positive_contrast_templates,
        positive_words, label=1, num_samples=num_per_class, seen=seen,
    )
    negative = _generate_text(
        negative_templates + negative_negation_templates
        + negative_contrast_templates + negative_sarcasm_templates,
        negative_words, label=0, num_samples=num_per_class, seen=seen,
    )

    all_data = positive + negative
    random.shuffle(all_data)

    out_path = Path(processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    stats = {
        "total": len(all_data),
        "positive": len(positive),
        "negative": len(negative),
        "unique": len({x["text"] for x in all_data}),
        "seed": seed,
        "path": str(out_path),
    }
    return stats


# ==================== Dataset ====================

def load_data(path: str | Path) -> tuple[list[str], list[int]]:
    """读取 [{text, label}, ...] 格式的 JSON，返回 (texts, labels)。"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [item["text"] for item in items], [item["label"] for item in items]


class SentimentDataset(torch.utils.data.Dataset):
    """把文本 + 标签包装成 Trainer 能用的数据集，__getitem__ 时做 tokenize。

    训练走的是 padding="max_length"（Trainer 要求定长），推理不走这里——
    推理在 model.py 里做动态 padding。
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(self.texts[idx], truncation=True,
                                  max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
