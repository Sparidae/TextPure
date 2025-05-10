import os
import threading
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_prompt

# 加载环境变量
load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE", "https://api.siliconflow.cn/v1")
model_name = os.getenv("LLM_FOR_CLEAN", "Qwen/Qwen3-30B-A3B")
context_length = int(os.getenv("LLM_FOR_CLEAN_CL", "4096"))

# 创建OpenAI客户端并使用锁保护它
client = OpenAI(api_key=API_KEY, base_url=API_BASE)
client_lock = threading.Lock()

# 请求速率限制相关
request_counter = 0
request_counter_lock = threading.Lock()
request_timestamp = time.time()
requests_per_minute = 1000  # 默认每分钟1000个请求


def completion(
    prompt,
    model=model_name,
    temperature=0.7,
    max_tokens=4096,
    max_retry_iters=3,
    retry_delays=1,
):
    """
    调用大模型接口获取完成结果 (使用requests库)

    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数
        max_retry_iters: 最大重试次数
        retry_delays: 重试间隔秒数

    Returns:
        (success, content): 是否成功及模型输出内容
    """
    # 构建请求
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    }

    # 重试逻辑
    for attempt in range(max_retry_iters):
        try:
            # 速率限制检查
            with request_counter_lock:
                global request_counter, request_timestamp
                current_time = time.time()
                # 如果已经过了一分钟，重置计数器
                if current_time - request_timestamp >= 60:
                    request_counter = 0
                    request_timestamp = current_time

                # 如果达到每分钟请求限制，等待
                if request_counter >= requests_per_minute:
                    wait_time = 60 - (current_time - request_timestamp)
                    if wait_time > 0:
                        time.sleep(wait_time)
                        # 重置计数器和时间戳
                        request_counter = 0
                        request_timestamp = time.time()

                # 增加请求计数
                request_counter += 1

            # 发送请求
            response = requests.request(
                "POST", url=f"{API_BASE}/chat/completions", json=payload, headers=headers
            )

            # 检查响应状态
            if response.status_code != 200:
                print(f"API请求失败: {response.status_code} - {response.text}")
                raise Exception(f"API请求失败，状态码: {response.status_code}")

            # 解析响应
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]

            # 打印token使用情况
            if "usage" in response_data and "completion_tokens" in response_data["usage"]:
                print(f"消耗token量: {response_data['usage']['completion_tokens']}")

            return True, content

        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {str(e)}")

            if attempt < max_retry_iters - 1:
                time.sleep(retry_delays)
            else:
                print(f"多次尝试后仍然失败，最后错误: {str(e)}")

    return False, None


def completion_with_openai(
    prompt,
    model=model_name,
    temperature=0.7,
    max_retry_iters=3,
    retry_delays=1,
):
    """
    调用大模型接口获取完成结果 (使用OpenAI官方库)

    Args:
        prompt: 提示词
        model: 模型名称
        temperature: 温度参数
        max_retry_iters: 最大重试次数
        retry_delays: 重试间隔秒数

    Returns:
        (success, content): 是否成功及模型输出内容
    """
    # 准备消息
    messages = [{"role": "user", "content": prompt}]

    # 重试逻辑
    for attempt in range(max_retry_iters):
        try:
            # 速率限制检查
            with request_counter_lock:
                global request_counter, request_timestamp
                current_time = time.time()
                # 如果已经过了一分钟，重置计数器
                if current_time - request_timestamp >= 60:
                    request_counter = 0
                    request_timestamp = current_time

                # 如果达到每分钟请求限制，等待
                if request_counter >= requests_per_minute:
                    wait_time = 60 - (current_time - request_timestamp)
                    if wait_time > 0:
                        time.sleep(wait_time)
                        # 重置计数器和时间戳
                        request_counter = 0
                        request_timestamp = time.time()

                # 增加请求计数
                request_counter += 1

            # 使用锁保护OpenAI客户端调用
            with client_lock:
                # 使用OpenAI客户端调用API
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024,
                    top_p=0.7,
                    stream=False,
                )

            # 获取响应内容
            content = response.choices[0].message.content

            # 打印token使用情况
            if hasattr(response, "usage") and hasattr(
                response.usage, "completion_tokens"
            ):
                print(f"消耗token量: {response.usage.completion_tokens}")

            return True, content

        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {str(e)}")

            if attempt < max_retry_iters - 1:
                time.sleep(retry_delays)
            else:
                print(f"多次尝试后仍然失败，最后错误: {str(e)}")

    return False, None


def get_completion(use_openai_client=False):
    """
    根据参数选择使用哪个完成函数

    Args:
        use_openai_client: 是否使用OpenAI客户端库

    Returns:
        选择的完成函数
    """
    if use_openai_client:
        return completion_with_openai
    else:
        return completion


if __name__ == "__main__":
    # 测试两种方法
    prompt = get_prompt(
        "rag",
        text="""
BanG Dream!十周年轨迹展海报上的十团主唱合影，最上一行左起：三角初华（Ave Mujica）、高松灯（MyGO!!!!!）、仲町阿拉蕾（梦限大MewType）；第二行左起：美竹兰（Afterglow）、凑友希那（Roselia）、仓田真白（Morfonica）、和奏瑞依（RAISE A SUILEN）；第三行左起：丸山彩（Pastel*Palettes）、户山香澄（Poppin'Party）、弦卷心（Hello, Happy World!）。
### 
以少女乐队为主题的MediaMix企划BanG Dream!，目前旗下共有10支主要乐队。乐队的活动范围涵盖了动画、漫画、游戏、小说、音乐、真实声优演出以及虚拟YouTuber活动等多个领域。 
按照其活动方式，可作出如下划分： 
  * Poppin'Party、Roselia等九团在以游戏《BanG Dream! 少女乐团派对！》为核心的一系列作品中登场，属同一世界观——（其中Ave Mujica未实装进入游戏但已在各作品中登场）。梦限大MewType作为虚拟YouTuber乐队，以直播活动和线下演出为主，尚未被统合至该世界观。
  * Poppin'Party、Roselia、RAISE A SUILEN、Morfonica、MyGO!!!!!、Ave Mujica、梦限大MewType七团声优（中之人）均可现场演奏，而俗称“手游团”或“虚拟团”的Afterglow、Pastel*Palettes和Hello, Happy World!则在多数情况下仅主唱演唱。


按照代际划分： 
  * Poppin'Party、Roselia、Afterglow、Pastel*Palettes、Hello, Happy World!作为游戏初期即实装的乐队，俗称“老五团”。
  * RAISE A SUILEN和Morfonica为后期追加的乐队，与老五团合称为“七大团”。
  * MyGO!!!!!、Ave Mujica和梦限大MewType为新兴乐队，早期即由新任总指挥根本雄贵主导，均具有独立社交媒体账号。

主要乐队的出场情况  乐队名  | Poppin'Party | Roselia | Afterglow | Pastel*Palettes | Hello, Happy World! | RAISE A SUILEN | Morfonica | MyGO!!!!! | Ave Mujica | 梦限大MewType  
---|---|---|---|---|---|---|---|---|---|---  
声优现场表演  | 演奏乐器并演唱  | 多数情况仅主唱演唱  | 演奏乐器并演唱   
动画  | 前三季动画登场  | 登场  | 第一季客串，后两季登场  | 第二季登场  | /  | 世界观尚未统一   
迷你动画系列  | PICO全系列登场  | PICO全系列及《Pastel Life》登场  | PICO全系列登场  | PICO大份起登场  | /   
FILM LIVE系列  | 全系列登场  | 一期客串，二期登场  | 二期登场  | /   
Roselia剧场版  | 登场或部分登场  | /   
Poppin'Dream!  | 登场  | /   
It's MyGO!!!!!&Ave Mujica  | 登场  | 部分登场  | 登场  | 部分登场  | /  | 部分登场  | 登场   
游戏  | 手游  | 初期可用角色  | 追加可用角色  | /   
Switch  | 追加可用角色（DLC）  | /   
### 
作为乐队的吉他和主唱，户山香澄一直在寻找着，如同小时候仰望星空时听到的“星之律动”一样的，闪闪发光而令人心动的事物。由此才促成了这支乐队的组建。总是把有咲家的仓库地下室当作练习场地的她们，技术有待提高，经验也不充分。乐队的成员除了香澄，还有钟爱音乐的天然少女花园多惠，希望能改变自己消极性格的牛込里美，温柔而关心家人的山吹沙绫，以及爱唱反调而心口不一的市谷有咲。在这个少女乐队时代中闪闪发光的女子高中生5人乐队组合，将自己的心情寄托于音乐当中，为了将更多让人心动不已的事情传递出去而每日歌唱。 
Poppin'Party 成员的姓氏均来自东京都新宿区地名。
Poppin'Party中四位成员名字的首字母S（沙绫Saaya)、T(多惠Tae)、A(有咲Arisa)、R(里美Rimi），组合起来是星辰。 再加上香澄Kasumi的首字母则是完全的。  
担当：吉他手、主唱  
乐器：ESP RANDOM STAR Kasumi  
年级：高中一年级→高中二年级→高中三年级  
生日：7月14日  
星座：巨蟹座  
喜欢的食物：炸薯条、白米饭  
讨厌的食物：纳豆  
兴趣：卡拉OK、冒险、想要尝试各种各样的事情  
行动力很好，有着积极乐观的性格。  
对待朋友全心全意，总是被许多朋友包围着。  
感性充沛，上了高中之后一直在寻找闪闪发光令人心动的东西，并将其唱成乐曲。  
有1个小1年的妹妹，但是一眼看过去却分不清谁是姐姐谁是妹妹。  
姓氏「户山」来自东京都新宿区户山町。 代表色：  
担当：主吉他手  
乐器：ESP SNAPPER Tae  
年级：高中一年级→高中二年级→高中三年级  
生日：12月4日  
星座：射手座  
喜欢的食物：汉堡肉（无比的肉食爱好者)、年糕红豆粥、能吃的都喜欢  
讨厌的食物：无（猎奇的东西之类的话...？）  
兴趣：跑步（不要勉强）、黏土、泡澡  
小学学习吉他的实力派。非常喜欢音乐，升入高中后就一直在Live House打工。  
在组成乐队前，常常都是自己一个人弹奏，但跟大家一起演奏时感到了震撼与乐趣。  
非常中意自己攒钱买来的蓝色吉他。  
有自己的步调并且相当天然，时不时会做出意料之外的行动吓到周围的人。  
家人是父亲母亲和20只兔子。  
姓氏「花园」来自东京都新宿区花园町。 代表色：  
担当：贝斯手  
乐器：ESP VIPER BASS Rimi  
年级：高中一年级→高中二年级→高中三年级  
生日：3月23日  
星座：白羊座  
喜欢的食物：巧克力、肉、鲜奶油  
讨厌的食物：巧克力薄荷味  
兴趣：游戏、读书  
初中时住在关西，情绪一激动就会不小心讲出关西腔。  
因为香澄在高中的开学典礼上毫无保留积极向上的自我介绍，开始对她感到在意。  
虽然想改变自己的胆小怯懦与毫无主见，但还是不能好好行动。  
山吹面包房的常客，最喜欢的东西是巧克力螺。  
姓氏「牛込」来自东京都新宿区牛込地区。 代表色：  
担当：鼓手  
乐器：Pearl Reference PURE Drum Kit  
年级：高中一年级→高中二年级→高中三年级  
生日：5月19日  
星座：金牛座  
喜欢的食物：peperoncino（一种意面）、奶酪  
讨厌的食物：生的海鲜  
兴趣：卡拉OK、观看棒球赛、收集发饰  
开学典礼就和香澄交好，经常和她一起吃饭，也是她的咨询对象。  
一边读高中一边在自家的面包店（山吹面包房）帮忙，是个孝顺的女儿。  
性格相当沉稳，又温柔地为朋友着想，是Poppin'Party里不可或缺的精神支柱。  
家中有年幼的弟弟和妹妹。跟以前所属的乐团成员现在关系也很良好。  
姓氏「山吹」来自东京都新宿区山吹町。 代表色：  
担当：键盘手  
乐器：Roland JUNO-DS61W、Roland AX-Synth  
年级：高中一年级→高中二年级→高中三年级  
生日：10月27日  
星座：天蝎座  
喜欢的食物：豆沙水果凉粉、玄米、白煮蛋  
讨厌的食物：葱类  
兴趣：盆栽、网上冲浪  
当铺“流星堂”的孙女，以盆景和上网为兴趣的宅，通过乐团活动逐渐拓展交际。  
虽然基本一直不出门，但在学校学生会里工作，有一套学习方法让自己成绩优异。  
非常毒舌，特别是总对香澄很强硬，但其实只是坦率不起来而已。  
在年幼时就学习过钢琴、不过半途而废了。  
姓氏「市谷」来自东京都新宿区市谷地区。 代表色：  
青梅竹马的五人组成的少女乐队。由于担心独自被分到不同班级的兰，为了能让五个人在一起，她们才组建了乐队。升入高中后，她们单纯地享受着乐队带来的快乐，放学后会在录音棚练习，还会参加演唱会。成员们的关系非常好，基本上没吵过架。表演本身虽然显得有些不修边幅，但充满力量的演奏和歌喉非常受欢迎。在各种不同的舞台上，她们在音乐中渡过“和往常一样”的每一天。  
Afterglow 成员的姓氏均来自东京都涩谷区地名。  
担当：主唱&吉他  
乐器：Gibson Les Paul Special SL Red  
年级：高中一年级→高中二年级→高中三年级  
生日：4月10日  
血型：A型  
星座：白羊座  
喜欢的东西： 苦味的点心  
讨厌的东西： 青豆  
兴趣：没有  
拥有百年历史的花道家族中的独生女。  
虽然被认为是个倔强又冷淡的人，但其实拥有比谁都要温热的一颗心。  
个性好强讨厌失败，也有着感到寂寞的一面。  
克服了跟父亲的冲突之后，对于花道也能正面看待了。  
非常重视青梅竹马和亲人。  
姓氏「美竹」来自东京都涩谷区美竹町。 代表色：
担当：主吉他手  
乐器：Schecter BH-1-STD-24  
年级：高中一年级→高中二年级→高中三年级  
生日：9月3日  
血型：B型  
星座：处女座  
喜欢的东西： 面包  
讨厌的东西： 辣的东西  
兴趣：收集点数卡、睡觉  
对于没兴趣的东西完全不在乎，而为了喜欢的人就能拼尽全力的类型。  
个性超级自我中心，说话语调十分懒散。  
喜欢对于一件事坚持到底，乐器便是其中之一。  
每天都弹奏着“感动人心”的音乐。  
对于面包有超乎寻常的热爱，是山吹面包房的常客。  
姓氏「青叶」来自东京都涩谷区青叶町。 代表色：  
CV：三泽纱千香（所属：STARDUST）""",
    )

    _, content = completion(prompt, model="deepseek-ai/DeepSeek-V3", max_tokens=8192)
    print(content)
