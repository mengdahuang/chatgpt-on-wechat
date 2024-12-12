# encoding:utf-8
import time
import openai
import openai.error
import requests
from common import memory, utils, const
from bot.bot import Bot
from bot.fastgpt.fastgpt_session import FastGPTSession
from bot.openai.open_ai_image import OpenAIImage
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from common.token_bucket import TokenBucket
from config import conf, load_config
from bot.baidu.baidu_wenxin_session import BaiduWenxinSession
from channel.wework.wework_channel import WeworkChannel
from channel.wechatcom.wechatcomapp_message import WechatComAppMessage
from wechatpy.enterprise import WeChatClient
from channel.wechatcom.wechatcomapp_channel import WechatComAppChannel
from wechatpy.exceptions import WeChatClientException

# OpenAI对话模型API (可用)
class FastGPTBot(Bot, OpenAIImage):
    def __init__(self):
        super().__init__()
        # set the default api_key
        openai.api_key = conf().get("open_ai_api_key")
        if conf().get("open_ai_api_base"):
            openai.api_base = conf().get("open_ai_api_base")
        proxy = conf().get("proxy")
        if proxy:
            openai.proxy = proxy
        if conf().get("rate_limit_chatgpt"):
            self.tb4chatgpt = TokenBucket(conf().get("rate_limit_chatgpt", 20))
        # 获取已有的 WeChatClient 实例
        self.wechat_client = WechatComAppChannel().client
        
        conf_model = conf().get("model") or "gpt-3.5-turbo"
        self.sessions = SessionManager(FastGPTSession, model=conf().get("model") or "gpt-3.5-turbo")
        # o1相关模型不支持system prompt，暂时用文心模型的session

        self.args = {
            "model": conf_model,  # 对话模型的名称
            "temperature": conf().get("temperature", 0.1),  # 值在[0,1]之间，越大表示回复越具有不确定性
            # "max_tokens":4096,  # 回复最大的字符数
            "top_p": conf().get("top_p", 1),
            "frequency_penalty": conf().get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "presence_penalty": conf().get("presence_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "request_timeout": conf().get("request_timeout", None),  # 请求超时时间，openai接口默认设置为600，对于难问题一般需要较长时间
            "timeout": conf().get("request_timeout", None),  # 重试超时时间，在这个时间内，将会自动重试
        }
        # o1相关模型固定了部分参数，暂时去掉
        if conf_model in [const.O1, const.O1_MINI]:
            self.sessions = SessionManager(BaiduWenxinSession, model=conf().get("model") or const.O1_MINI)
            remove_keys = ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
            for key in remove_keys:
                self.args.pop(key, None)  # 如果键不存在，使用 None 来避免抛出错误

    def get_uname(self, context):
        msg = context.kwargs['msg']
        if context.kwargs['isgroup']:
            uname = msg.actual_user_nickname
        else:
            if isinstance(msg, WechatComAppMessage):
                uname = context["receiver"]
            else:
                uname = msg.from_user_nickname
        return uname

    def reply(self, query, context=None):
        # acquire reply content
        if context.type == ContextType.TEXT:
            # logger.info("[CHATGPT] query={}".format(query))
            # 2024/10/24 change to below
            # 获取用户信息
            uname = self.get_uname(context)
            logger.info("[user]{}, <<USER_QUESTION_START>> query={} <<USER_QUESTION_END>>".format(uname, query))
            try:
                user_info = self.wechat_client.user.get(uname)
                logger.info(f"获取到的用户信息：{user_info}")
            except WeChatClientException as e:
                logger.error(f"无法获取用户信息：{e}")
                user_info = None

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["#清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "#清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "#更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[CHATGPT] session query={}".format(session.messages))

            api_key = context.get("openai_api_key")
            model = context.get("gpt_model")
            new_args = None
            if model:
                new_args = self.args.copy()
                new_args["model"] = model
            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, session_id)

            reply_content = self.reply_text(session, api_key, args=new_args)
            logger.debug(
                "[CHATGPT] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[CHATGPT] reply {} used 0 tokens.".format(reply_content))
            return reply

        elif context.type == ContextType.IMAGE_CREATE:
            ok, retstring = self.create_img(query, 0)
            reply = None
            if ok:
                reply = Reply(ReplyType.IMAGE_URL, retstring)
            else:
                reply = Reply(ReplyType.ERROR, retstring)
            return reply
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def reply_text(self, session: FastGPTSession, context, api_key=None, args=None, retry_count=0) -> dict:
        """
        call openai's ChatCompletion to get the answer
        :param session: a conversation session
        :param session_id: session id
        :param retry_count: retry count
        :return: {}
        """
        try:
            if conf().get("rate_limit_chatgpt") and not self.tb4chatgpt.get_token():
                raise openai.error.RateLimitError("RateLimitError: rate limit exceeded")
            # if api_key == None, the default openai.api_key will be used
            if args is None:
                args = self.args
            # 为fastgpt增加用户相关字段
            # 在 args 中添加 chatId（即 session_id）
            msg = context['msg']
            # args['chatId'] = session.session_id  # 添加 chatId
            
            if context['isgroup']:
                uname = context['msg'].actual_user_nickname
                roomName = context['msg'].other_user_nickname
                uid = context["msg"].from_user_id
            else:
                if isinstance(msg, WechatComAppMessage):
                    uname = context["receiver"]  # 私聊中，如果是 WechatComAppMessage，则发送者设为 context["receiver"]
                    uid = ""
                else:
                    uname = context['msg'].from_user_nickname
                    uid = context["msg"].from_user_id
                roomName = ""            
            args['variables'] = {
                "uid" : uid,  # 发件人ID
                "uname":uname,  # 私聊发件人
                # "uname": context["msg"].actual_user_id,
                # "ualias": context["msg"].actual_user_nickname,  # 群聊发件人
                "roomName": roomName, # 群
            }
            # 为fastgpt增加chatId
            response = openai.ChatCompletion.create(api_key=api_key, messages=session.messages, chatId=session.session_id, **args)
            # response = openai.ChatCompletion.create(api_key=api_key, messages=session.messages, **args)
            # logger.debug("[CHATGPT] response={}".format(response))
            # logger.info("[ChatGPT] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))
            return {
                "total_tokens": response["usage"]["total_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "content": response.choices[0]["message"]["content"],
            }
        except Exception as e:
            status_url = "https://status.cloudosd.com/status/service-ait"
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "服务异常，请稍后再试。\n当前AiT状态【异常】，实时状态查询: "+ status_url}
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[CHATGPT] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请稍后再试。\n错误信息：{}\n当前AiT状态【异常】，实时状态查询: {}".format(error_message, status_url)
                if need_retry:
                    time.sleep(2)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[CHATGPT] Timeout: {}".format(e))
                result["content"] = "会话超时，请稍后再试。\n错误信息：{}\n当前AiT状态【异常】，实时状态查询: {}".format(error_message, status_url)
                if need_retry:
                    time.sleep(2)
            elif isinstance(e, openai.error.APIError):
                logger.warn("[CHATGPT] Bad Gateway: {}".format(e))
                result["content"] = "API请求出错，请稍后再试。\n错误信息：{}\n当前AiT状态【异常】，实时状态查询: {}".format(error_message, status_url)
                if need_retry:
                    time.sleep(2)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[CHATGPT] APIConnectionError: {}".format(e))
                result["content"] = "API网络连接异常，请稍后再试。\n错误信息：{}\n当前AiT状态【异常】，实时状态查询: {}".format(error_message, status_url)
                if need_retry:
                    time.sleep(2)
            else:
                logger.exception("[CHATGPT] Exception: {}".format(e))
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[CHATGPT] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, api_key, args, retry_count + 1)
            else:
                return result