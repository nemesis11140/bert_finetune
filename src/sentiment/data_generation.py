"""数据生成：基于模板 + 词库合成情感标注句子。

模板和词库都在本文件里，想加新的句式/词汇直接在这里扩充。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

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

FOOD = ["红烧肉", "水煮鱼", "糖醋排骨", "麻婆豆腐", "宫保鸡丁", "炒时蔬", "汤", "甜点", "烤鸭", "火锅",
        "酸菜鱼", "小龙虾", "烧烤", "生煎", "牛肉面", "卤味", "蒸鱼", "虾饺"]
ACTION_POS = ["成功", "完成", "实现", "达成", "做到了", "实现了目标", "通过了", "上岸了", "拿下了", "搞定了"]
ACTION_NEG = ["坏了", "卡顿", "崩溃", "死机", "闪退", "关机了", "没电了", "出问题了", "黑屏", "断网",
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
        positive_templates + positive_negation_templates,
        positive_words, label=1, num_samples=num_per_class, seen=seen,
    )
    negative = _generate_text(
        negative_templates + negative_negation_templates,
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
