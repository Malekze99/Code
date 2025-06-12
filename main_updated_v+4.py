#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import logging
import os
import re
import time
import json # Added for JSON operations
from decimal import Decimal, ROUND_DOWN, InvalidOperation, Context as DecimalContext, setcontext
from datetime import datetime # Added for timestamps

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

# Set higher precision for Decimal context
setcontext(DecimalContext(prec=30))

# إعداد السجل (Logging)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- إعدادات الملف --- #
CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "main_config.ini")
TESTNET_TRADES_FILE = "testnet_trades.json" # Define JSON file path

def load_config():
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.error(f"ملف الإعدادات {CONFIG_FILE_PATH} غير موجود. جاري إنشاء ملف افتراضي.")
        create_default_config()
        # Reload after creating default
        config.read(CONFIG_FILE_PATH, encoding="utf-8")
    else:
        try:
            config.read(CONFIG_FILE_PATH, encoding="utf-8")
        except Exception as e:
            logger.error(f"خطأ في قراءة ملف الإعدادات {CONFIG_FILE_PATH}: {e}")
            return None

    if not config.sections():
        logger.error(f"ملف الإعدادات {CONFIG_FILE_PATH} فارغ أو غير صالح.")
        return None
    return config

def create_default_config():
    logger.info(f"جاري إنشاء ملف إعدادات افتراضي في {CONFIG_FILE_PATH}")
    config = configparser.ConfigParser()
    config["TELEGRAM"] = {"TOKEN": "YOUR_TELEGRAM_BOT_TOKEN_HERE"}
    config["BINANCE"] = {
        "TESTNET_API_KEY": "YOUR_BINANCE_TESTNET_API_KEY_HERE",
        "TESTNET_SECRET_KEY": "YOUR_BINANCE_TESTNET_SECRET_KEY_HERE",
        "LIVE_API_KEY": "YOUR_BINANCE_LIVE_API_KEY_HERE",
        "LIVE_SECRET_KEY": "YOUR_BINANCE_LIVE_SECRET_KEY_HERE",
        "DEFAULT_BINANCE_MODE": "testnet"
    }
    config["TRADING"] = {
        "DEFAULT_USDT_AMOUNT_PER_TRADE_AUTO": "50",
        "DEFAULT_PROFIT_PERCENTAGE_AUTO": "5",
        "INITIAL_TESTNET_USDT_BALANCE": "10000" # Added initial balance config
    }
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as configfile:
            config.write(configfile)
        logger.info(f"تم إنشاء ملف إعدادات افتراضي. عدل {CONFIG_FILE_PATH} بالمفاتيح الحقيقية.")
    except Exception as e:
        logger.error(f"فشل كتابة ملف الإعدادات الافتراضي: {e}")

config = load_config()

# تحميل رمز Telegram
TELEGRAM_BOT_TOKEN = None
if config and config.has_section("TELEGRAM") and config.has_option("TELEGRAM", "TOKEN"):
    TELEGRAM_BOT_TOKEN = config["TELEGRAM"]["TOKEN"]
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE" or not TELEGRAM_BOT_TOKEN:
        logger.error(f"رمز بوت Telegram غير موجود أو افتراضي في {CONFIG_FILE_PATH}.")
        TELEGRAM_BOT_TOKEN = None
else:
    logger.error(f"إعدادات رمز Telegram غير موجودة في {CONFIG_FILE_PATH}")

# --- Testnet Trade Storage --- START
def load_testnet_trades(bot_data: dict) -> None:
    """Loads testnet trades and simulated balance from JSON file."""
    try:
        if os.path.exists(TESTNET_TRADES_FILE):
            with open(TESTNET_TRADES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Load trades, converting relevant fields back to Decimal
                trades = data.get("trades", {})
                loaded_trades = {}
                for symbol, trade_info in trades.items():
                    try:
                        loaded_trades[symbol] = {
                            "quantity": Decimal(trade_info.get("quantity", "0")),
                            "avg_price": Decimal(trade_info.get("avg_price", "0")),
                            "buy_time": trade_info.get("buy_time", time.time()), # Keep as float timestamp
                            "cost_usdt": Decimal(trade_info.get("cost_usdt", "0")) # Load cost
                        }
                    except (InvalidOperation, TypeError) as dec_err:
                        logger.error(f"خطأ في تحويل بيانات الصفقة المحملة لـ {symbol}: {dec_err}. تخطي الصفقة.")
                bot_data["open_trades_data"] = loaded_trades
                # Load simulated balance
                bot_data["simulated_usdt_balance"] = Decimal(data.get("simulated_usdt_balance", config["TRADING"].get("INITIAL_TESTNET_USDT_BALANCE", "10000")))
                logger.info(f"تم تحميل {len(loaded_trades)} صفقة تجريبية ورصيد USDT محاكى ({bot_data["simulated_usdt_balance"]}) من {TESTNET_TRADES_FILE}")
        else:
            logger.info(f"ملف {TESTNET_TRADES_FILE} غير موجود. بدء بصفقات فارغة ورصيد افتراضي.")
            bot_data["open_trades_data"] = {}
            bot_data["simulated_usdt_balance"] = Decimal(config["TRADING"].get("INITIAL_TESTNET_USDT_BALANCE", "10000"))
    except json.JSONDecodeError as e:
        logger.error(f"خطأ في فك تشفير JSON من {TESTNET_TRADES_FILE}: {e}. بدء بصفقات فارغة.")
        bot_data["open_trades_data"] = {}
        bot_data["simulated_usdt_balance"] = Decimal(config["TRADING"].get("INITIAL_TESTNET_USDT_BALANCE", "10000"))
    except Exception as e:
        logger.error(f"خطأ غير متوقع أثناء تحميل الصفقات التجريبية: {e}")
        bot_data["open_trades_data"] = {}
        bot_data["simulated_usdt_balance"] = Decimal(config["TRADING"].get("INITIAL_TESTNET_USDT_BALANCE", "10000"))

def save_testnet_trades(bot_data: dict) -> None:
    """Saves current testnet trades and simulated balance to JSON file."""
    try:
        trades_to_save = {}
        open_trades = bot_data.get("open_trades_data", {})
        for symbol, trade_info in open_trades.items():
            # Convert Decimals to strings for JSON serialization
            trades_to_save[symbol] = {
                "quantity": str(trade_info.get("quantity", Decimal("0"))),
                "avg_price": str(trade_info.get("avg_price", Decimal("0"))),
                "buy_time": trade_info.get("buy_time", time.time()),
                "cost_usdt": str(trade_info.get("cost_usdt", Decimal("0"))) # Save cost
            }

        data_to_save = {
            "trades": trades_to_save,
            "simulated_usdt_balance": str(bot_data.get("simulated_usdt_balance", Decimal("0")))
        }
        with open(TESTNET_TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        logger.debug(f"تم حفظ الصفقات التجريبية والرصيد المحاكى في {TESTNET_TRADES_FILE}")
    except Exception as e:
        logger.error(f"فشل حفظ الصفقات التجريبية في {TESTNET_TRADES_FILE}: {e}")
# --- Testnet Trade Storage --- END

# --- تكامل Binance --- #
def initialize_binance_client(context: ContextTypes.DEFAULT_TYPE | Application) -> Client | None:
    bot_data = context.bot_data
    if not config or not config.has_section("BINANCE"):
        logger.error("قسم إعدادات Binance غير موجود في ملف الإعدادات.")
        bot_data["binance_client"] = None
        return None

    current_mode = bot_data.get("binance_mode", "testnet")
    logger.info(f"جاري تهيئة عميل Binance في وضع {current_mode.upper()}.")

    api_key = None
    secret_key = None
    is_testnet = (current_mode == "testnet")

    if is_testnet:
        api_key = config["BINANCE"].get("TESTNET_API_KEY")
        secret_key = config["BINANCE"].get("TESTNET_SECRET_KEY")
        if not api_key or api_key == "YOUR_BINANCE_TESTNET_API_KEY_HERE":
            logger.error("مفتاح API لـ Testnet غير موجود أو افتراضي.")
            api_key = None
        if not secret_key or secret_key == "YOUR_BINANCE_TESTNET_SECRET_KEY_HERE":
            logger.error("المفتاح السري لـ Testnet غير موجود أو افتراضي.")
            secret_key = None
    else: # live mode
        api_key = config["BINANCE"].get("LIVE_API_KEY")
        secret_key = config["BINANCE"].get("LIVE_SECRET_KEY")
        if not api_key or api_key == "YOUR_BINANCE_LIVE_API_KEY_HERE":
            logger.warning("مفتاح API لـ Live غير موجود أو افتراضي.")
            api_key = None
        if not secret_key or secret_key == "YOUR_BINANCE_LIVE_SECRET_KEY_HERE":
            logger.warning("المفتاح السري لـ Live غير موجود أو افتراضي.")
            secret_key = None

    client = None
    if api_key and secret_key:
        try:
            client = Client(api_key, secret_key, tld="com", testnet=is_testnet)
            client.ping()
            logger.info(f"تم الاتصال بـ Binance API (وضع {"Testnet" if is_testnet else "Live"}) بنجاح.")
            # Cache symbol info after successful connection
            try:
                exchange_info = client.get_exchange_info()
                bot_data["exchange_info"] = {s["symbol"]: s for s in exchange_info["symbols"]}
                logger.info(f"تم تخزين معلومات {len(bot_data["exchange_info"])} رمز مؤقتًا.")
            except Exception as ex_info_err:
                logger.error(f"فشل جلب وتخزين معلومات الرموز: {ex_info_err}")
                bot_data["exchange_info"] = {}
        except BinanceAPIException as e:
            logger.error(f"خطأ API من Binance أثناء تهيئة العميل ({current_mode}): {e}")
            client = None
            bot_data["exchange_info"] = {}
        except Exception as e:
            logger.error(f"فشل تهيئة عميل Binance ({current_mode}): {e}")
            client = None
            bot_data["exchange_info"] = {}
    else:
        logger.warning(f"مفتاح API أو المفتاح السري مفقود لوضع {current_mode.upper()}.")
        bot_data["exchange_info"] = {}

    bot_data["binance_client"] = client
    return client

# --- دوال مساعدة لـ Binance --- #
def get_symbol_info_from_cache(context: ContextTypes.DEFAULT_TYPE, symbol: str) -> dict | None:
    """Gets symbol information from the cached exchange info."""
    exchange_info_cache = context.bot_data.get("exchange_info", {})
    return exchange_info_cache.get(symbol)

def validate_symbol(client: Client, context: ContextTypes.DEFAULT_TYPE, symbol: str) -> bool:
    """Validates symbol using cached info first, then API call as fallback."""
    symbol_info = get_symbol_info_from_cache(context, symbol)
    if symbol_info:
        return True
    # Fallback to API call if not in cache or cache is empty
    logger.warning(f"الرمز {symbol} غير موجود في الذاكرة المؤقتة، محاولة جلب من API.")
    if not client:
        logger.warning("لا يمكن التحقق من الرمز، عميل Binance غير مهيأ.")
        return False
    try:
        info = client.get_symbol_info(symbol)
        # Optionally update cache here if needed
        return bool(info)
    except BinanceAPIException as e:
        if e.code == -1121: # Invalid symbol
            logger.warning(f"رمز غير صالح {symbol}: {e}")
            return False
        logger.error(f"خطأ API Binance عند التحقق من الرمز {symbol}: {e}")
        return False # Treat API errors as invalid for safety
    except Exception as e:
        logger.error(f"خطأ غير متوقع في التحقق من الرمز {symbol}: {e}")
        return False

# --- LOT_SIZE Handling --- START
def get_symbol_filters(context: ContextTypes.DEFAULT_TYPE, symbol: str) -> tuple[int, int, Decimal, Decimal]:
    """Gets quantity precision, price precision, minQty, and stepSize for a symbol."""
    symbol_info = get_symbol_info_from_cache(context, symbol)
    if not symbol_info:
        logger.warning(f"لم يتم العثور على معلومات الرمز {symbol} في الذاكرة المؤقتة. استخدام قيم افتراضية.")
        # Attempt API call as fallback? For now, return defaults.
        return 8, 8, Decimal("0.00000001"), Decimal("0.00000001") # Default safe values

    qty_precision = int(symbol_info.get("baseAssetPrecision", symbol_info.get("quantityPrecision", 8)))
    price_precision = int(symbol_info.get("quotePrecision", symbol_info.get("pricePrecision", 8)))

    min_qty = Decimal("0")
    step_size = Decimal("1") # Default step size if filter not found

    # Find LOT_SIZE filter
    lot_size_filter = next((f for f in symbol_info.get("filters", []) if f.get("filterType") == "LOT_SIZE"), None)
    if lot_size_filter:
        min_qty = Decimal(lot_size_filter.get("minQty", "0"))
        step_size = Decimal(lot_size_filter.get("stepSize", "1"))
        logger.debug(f"مرشحات LOT_SIZE لـ {symbol}: minQty={min_qty}, stepSize={step_size}")
    else:
        logger.warning(f"لم يتم العثور على مرشح LOT_SIZE لـ {symbol}. استخدام قيم افتراضية.")

    # Ensure step_size is not zero to avoid division errors
    if step_size <= 0:
        logger.error(f"خطأ: stepSize لـ {symbol} هو صفر أو أقل. استخدام 1 كقيمة افتراضية.")
        step_size = Decimal("1")

    return qty_precision, price_precision, min_qty, step_size

def adjust_quantity_for_lot_size(quantity: Decimal, min_qty: Decimal, step_size: Decimal) -> Decimal | None:
    """Adjusts the quantity based on minQty and stepSize rules."""
    if quantity < min_qty:
        logger.warning(f"الكمية المحسوبة {quantity} أقل من الحد الأدنى {min_qty}. لا يمكن تنفيذ الأمر.")
        return None # Return None if below minimum

    # Adjust quantity based on step_size
    # Formula: floor(quantity / step_size) * step_size
    if step_size > 0:
        adjusted_quantity = (quantity // step_size) * step_size
        logger.debug(f"تعديل الكمية: الأصلية={quantity}, stepSize={step_size}, المعدلة={adjusted_quantity}")
        # Final check if adjusted quantity is still >= min_qty
        if adjusted_quantity < min_qty:
             logger.warning(f"الكمية المعدلة {adjusted_quantity} أصبحت أقل من الحد الأدنى {min_qty}. لا يمكن تنفيذ الأمر.")
             return None
        return adjusted_quantity
    else:
        # Should not happen due to check in get_symbol_filters, but handle defensively
        logger.error("خطأ: step_size هو صفر أثناء تعديل الكمية.")
        return quantity # Return original quantity if step_size is invalid
# --- LOT_SIZE Handling --- END

def format_decimal(number_str: str | Decimal | float, precision: int) -> str:
    try:
        num_decimal = Decimal(str(number_str))
        if precision < 0:
            logger.warning(f"الدقة لا يمكن أن تكون سالبة: {precision}. استخدام 0.")
            precision = 0
        # Create the quantizer string like '0.0001' for precision 4
        quantizer_str = '1e-' + str(precision)
        quantizer = Decimal(quantizer_str)
        formatted_decimal = num_decimal.quantize(quantizer, rounding=ROUND_DOWN)
        # Format to ensure correct number of decimal places, avoiding scientific notation
        return f"{formatted_decimal:.{precision}f}"
    except (InvalidOperation, TypeError, ValueError) as e:
        logger.error(f"تنسيق رقم غير صالح للتحويل إلى عشري: {number_str} ({type(number_str)}). Error: {e}")
        return f"{Decimal('0'):.{precision}f}"
    except Exception as e:
        logger.exception(f"خطأ في تنسيق العشري {number_str} بدقة {precision}: {e}")
        return f"{Decimal('0'):.{precision}f}"

def get_usdt_balance(client: Client) -> Decimal:
    if not client:
        logger.warning("لا يمكن جلب رصيد USDT، عميل Binance غير مهيأ.")
        return Decimal("0")
    try:
        balance = client.get_asset_balance(asset="USDT")
        logger.debug(f"تم جلب رصيد USDT بنجاح: {balance['free']}")
        return Decimal(balance["free"])
    except BinanceAPIException as e:
        logger.error(f"خطأ API من Binance في جلب رصيد USDT: {e}")
        return Decimal("0") # Return 0 on error
    except Exception as e:
        logger.error(f"خطأ في جلب رصيد USDT: {e}")
        return Decimal("0") # Return 0 on error

async def place_take_profit_order(client: Client, context: ContextTypes.DEFAULT_TYPE, symbol: str, quantity: Decimal, avg_buy_price: Decimal,
                                profit_percent: Decimal) -> dict | None:
    if not client:
        logger.warning(f"لا يمكن وضع أمر TP لـ {symbol}، عميل Binance غير مهيأ.")
        return None
    try:
        # Get precision and filters
        qty_precision, price_precision, min_qty, step_size = get_symbol_filters(context, symbol)

        # Adjust quantity for TP order
        adjusted_quantity_tp = adjust_quantity_for_lot_size(quantity, min_qty, step_size)
        if adjusted_quantity_tp is None or adjusted_quantity_tp <= 0:
            logger.error(f"فشل تعديل كمية أمر TP لـ {symbol} ({quantity}).")
            return None

        target_sell_price = avg_buy_price * (Decimal("1") + profit_percent / Decimal("100"))
        formatted_target_price = format_decimal(target_sell_price, price_precision)
        formatted_quantity_tp = format_decimal(adjusted_quantity_tp, qty_precision)

        logger.info(f"محاولة وضع أمر بيع TP لـ {symbol}: الكمية={formatted_quantity_tp}, السعر={formatted_target_price}")
        tp_order = client.order_limit_sell(
            symbol=symbol,
            quantity=formatted_quantity_tp,
            price=formatted_target_price
        )
        logger.info(f"تم وضع أمر جني أرباح لـ {symbol}: {tp_order}")
        return tp_order
    except BinanceOrderException as boe:
        logger.error(f"فشل وضع أمر جني أرباح لـ {symbol}: {boe}")
        return None
    except BinanceAPIException as bae:
        logger.error(f"خطأ API Binance عند وضع أمر TP لـ {symbol}: {bae}")
        return None
    except Exception as e:
        logger.error(f"خطأ غير متوقع في وضع أمر جني أرباح لـ {symbol}: {e}")
        return None

# --- معالجات بوت Telegram --- #
async def post_init(application: Application) -> None:
    if config:
        default_binance_mode = config["BINANCE"].get("DEFAULT_BINANCE_MODE", "testnet").lower()
        if default_binance_mode not in ["live", "testnet"]:
            logger.warning(f"وضع DEFAULT_BINANCE_MODE غير صالح '{default_binance_mode}'. استخدام testnet افتراضيًا.")
            default_binance_mode = "testnet"
        application.bot_data["binance_mode"] = default_binance_mode
        application.bot_data["trading_active"] = True
        application.bot_data["trading_mode"] = "off" # 'manual', 'auto', 'off'
        application.bot_data["auto_trade_amount_usdt"] = Decimal(config["TRADING"].get("DEFAULT_USDT_AMOUNT_PER_TRADE_AUTO", "50"))
        application.bot_data["auto_trade_profit_percent"] = Decimal(config["TRADING"].get("DEFAULT_PROFIT_PERCENTAGE_AUTO", "5"))
        # Initialize with empty dicts/defaults, load_testnet_trades will populate if needed
        application.bot_data["open_trades_data"] = {}
        application.bot_data["simulated_usdt_balance"] = Decimal(config["TRADING"].get("INITIAL_TESTNET_USDT_BALANCE", "10000"))
        application.bot_data["exchange_info"] = {} # Initialize cache

        logger.info(f"البوت يبدأ بوضع Binance افتراضي: {application.bot_data['binance_mode'].upper()}")
        initialize_binance_client(application)

        # Load testnet trades if starting in testnet mode
        if application.bot_data["binance_mode"] == "testnet":
            load_testnet_trades(application.bot_data)
    else:
        logger.error("لم يتم تحميل الإعدادات، لا يمكن تهيئة حالة البوت.")
        # Set defaults even if config fails
        application.bot_data["binance_mode"] = "testnet"
        application.bot_data["trading_active"] = False
        application.bot_data["trading_mode"] = "off"
        application.bot_data["open_trades_data"] = {}
        application.bot_data["simulated_usdt_balance"] = Decimal("0")
        application.bot_data["exchange_info"] = {}

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    # Ensure defaults are set if post_init failed or was skipped
    if not config:
        context.bot_data.setdefault("binance_mode", "testnet")
        context.bot_data.setdefault("trading_active", False)
        context.bot_data.setdefault("trading_mode", "off")
        context.bot_data.setdefault("open_trades_data", {})
        context.bot_data.setdefault("simulated_usdt_balance", Decimal("0"))
        context.bot_data.setdefault("exchange_info", {})
        await update.message.reply_html(
            rf"مرحباً {user.mention_html()}! حدث خطأ في تحميل الإعدادات. البوت غير نشط."
        )
        return

    default_mode_from_config = config["BINANCE"].get("DEFAULT_BINANCE_MODE", "testnet").lower()
    context.bot_data.setdefault("binance_mode", default_mode_from_config)
    context.bot_data.setdefault("trading_active", True)
    context.bot_data.setdefault("trading_mode", "off")
    context.bot_data.setdefault("auto_trade_amount_usdt", Decimal(config["TRADING"].get("DEFAULT_USDT_AMOUNT_PER_TRADE_AUTO", "50")))
    context.bot_data.setdefault("auto_trade_profit_percent", Decimal(config["TRADING"].get("DEFAULT_PROFIT_PERCENTAGE_AUTO", "5")))
    # Initialize, load_testnet_trades will overwrite if needed
    context.bot_data.setdefault("open_trades_data", {})
    context.bot_data.setdefault("simulated_usdt_balance", Decimal(config["TRADING"].get("INITIAL_TESTNET_USDT_BALANCE", "10000")))
    context.bot_data.setdefault("exchange_info", {})

    # Initialize client if not present or cache is empty
    if "binance_client" not in context.bot_data or \
       context.bot_data.get("binance_client") is None or \
       not context.bot_data.get("exchange_info"):
        logger.info("إعادة تهيئة عميل Binance و/أو تحديث معلومات الرموز عند أمر /start")
        initialize_binance_client(context)

    # Load testnet trades if in testnet mode and trades haven't been loaded yet (e.g., after restart)
    if context.bot_data["binance_mode"] == "testnet" and not context.bot_data.get("open_trades_data"): # Check if empty
        logger.info("تحميل بيانات Testnet عند أمر /start")
        load_testnet_trades(context.bot_data)

    current_binance_mode = context.bot_data.get("binance_mode", "N/A").upper()
    current_trading_mode = context.bot_data.get("trading_mode", "off")
    await update.message.reply_html(
        rf"مرحباً {user.mention_html()}! أنا بوت التداول الخاص بك. "
        rf"وضع Binance الحالي: {current_binance_mode}. "
        rf"وضع التداول: {current_trading_mode}. استخدم /help لعرض الأوامر المتاحة."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "الأوامر المتاحة:\n"
        "/start - بدء التفاعل مع البوت\n"
        "/help - عرض هذه الرسالة\n"
        "/status - عرض حالة البوت\n"
        "/switch_mode <live|testnet> - التبديل بين وضع Binance الحقيقي والتجريبي\n"
        "/balance - عرض رصيد USDT المتاح (حقيقي أو محاكى)\n"
        "/manual_mode_on - تفعيل وضع التداول اليدوي\n"
        "/manual_mode_off - إيقاف وضع التداول اليدوي\n"
        "/auto_mode_on [مبلغ_USDT] [نسبة_الربح%] - تفعيل التداول الآلي\n"
        "/auto_mode_off - إيقاف التداول الآلي\n"
        "/buy <الرمز> <مبلغ_USDT> [نسبة_الربح%] - شراء يدوي\n"
        "/sell <الرمز> [الكمية] - (تجريبي فقط) بيع محاكى لصفقة مفتوحة (بيع الكل إذا لم تحدد كمية)\n"
        "/stop_trading - إيقاف مؤقت للصفقات الجديدة\n"
        "/start_trading - استئناف فتح صفقات جديدة\n"
        "/open_trades - عرض الصفقات/الأرصدة المفتوحة (حقيقي: أرصدة الحساب, تجريبي: صفقات محاكاة)\n"
        "/profit_summary - (قيد التطوير) عرض ملخص الأرباح\n\n"
        "للتداول الآلي، أعد توجيه رسالة التوصية. للتداول اليدوي، استخدم /buy أو /sell (في Testnet)."
    )
    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    trading_mode = context.bot_data.get("trading_mode", "off")
    binance_mode = context.bot_data.get("binance_mode", "N/A").upper()
    is_active = context.bot_data.get("trading_active", True)
    auto_amount = context.bot_data.get("auto_trade_amount_usdt", Decimal("0"))
    auto_profit = context.bot_data.get("auto_trade_profit_percent", Decimal("0"))
    insufficient_balance = context.bot_data.get("insufficient_balance_auto_mode", False)

    status_text = f"حالة البوت:\n"
    status_text += f"وضع Binance: {binance_mode}\n"
    status_text += f"حالة التداول: {"نشط" if is_active else "متوقف مؤقتاً (/start_trading للاستئناف)"}\n"
    status_text += f"وضع التداول الحالي: {trading_mode}\n"

    if trading_mode == "auto":
        status_text += f"  مبلغ الصفقة الآلية: {format_decimal(auto_amount, 2)} USDT\n"
        status_text += f"  نسبة الربح الآلية: {format_decimal(auto_profit, 2)}%\n"
        if insufficient_balance:
            status_text += "  تنبيه: الرصيد غير كافٍ للتداول الآلي.\n"

    client = context.bot_data.get("binance_client")
    connection_status = "غير متصل ❌"
    if client:
        try:
            client.ping()
            connection_status = "متصل ✅"
            # Refresh exchange info cache if connection is ok but cache is empty
            if not context.bot_data.get("exchange_info"):
                logger.info("تحديث معلومات الرموز عند فحص الحالة.")
                try:
                    exchange_info = client.get_exchange_info()
                    context.bot_data["exchange_info"] = {s["symbol"]: s for s in exchange_info["symbols"]}
                    logger.info(f"تم تحديث معلومات {len(context.bot_data["exchange_info"])} رمز.")
                except Exception as ex_info_err:
                    logger.error(f"فشل تحديث معلومات الرموز عند فحص الحالة: {ex_info_err}")
                    context.bot_data["exchange_info"] = {}
        except Exception:
            connection_status = "فشل الاتصال ⚠️"
            context.bot_data["exchange_info"] = {} # Clear cache on connection failure

    status_text += f"اتصال Binance ({binance_mode}): {connection_status}\n"

    await update.message.reply_text(status_text)

async def switch_mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or len(context.args) != 1:
        await update.message.reply_text("الاستخدام: /switch_mode <live|testnet>")
        return

    new_mode = context.args[0].lower()
    if new_mode not in ["live", "testnet"]:
        await update.message.reply_text("وضع غير صالح. استخدم 'live' أو 'testnet'.")
        return

    current_mode = context.bot_data.get("binance_mode", "testnet")
    if new_mode == current_mode:
        await update.message.reply_text(f"البوت يعمل بالفعل في وضع {current_mode.upper()}.")
        return

    logger.info(f"المستخدم {update.effective_user.id} طلب التبديل إلى وضع {new_mode.upper()}.")
    context.bot_data["binance_mode"] = new_mode

    # Reset specific states when switching modes
    context.bot_data.pop("insufficient_balance_auto_mode", None)
    # Clear client and cache before initializing new one
    context.bot_data["binance_client"] = None
    context.bot_data["exchange_info"] = {}

    await update.message.reply_text(f"⏳ جاري التبديل إلى وضع {new_mode.upper()}...")
    new_client = initialize_binance_client(context)

    if new_client:
        await update.message.reply_text(f"✅ تم التبديل إلى وضع {new_mode.upper()}.")
        # Load testnet trades if switching TO testnet
        if new_mode == "testnet":
            load_testnet_trades(context.bot_data)
        else: # Clear testnet data if switching TO live
             context.bot_data["open_trades_data"] = {}
             # Keep simulated balance? Or reset? Let's keep it for now.
    else:
        await update.message.reply_text(f"❌ فشل الاتصال بـ Binance API. تحقق من مفاتيح {new_mode.upper()}.")
        # Revert mode on failure?
        # context.bot_data["binance_mode"] = current_mode
        # initialize_binance_client(context) # Re-init old client

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("عذراً، هذا الأمر غير معروف. استخدم /help لعرض الأوامر.")

async def get_usdt_balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    client = context.bot_data.get("binance_client")
    binance_mode = context.bot_data.get("binance_mode", "testnet")

    # --- Handle Testnet Mode ---
    if binance_mode == "testnet":
        # Ensure balance is loaded if not already
        if "simulated_usdt_balance" not in context.bot_data:
             load_testnet_trades(context.bot_data)
        simulated_balance = context.bot_data.get("simulated_usdt_balance", Decimal("0"))
        await update.message.reply_text(f"رصيد USDT المحاكى (Testnet): {format_decimal(simulated_balance, 2)}")
        return

    # --- Handle Live Mode --- 
    if not client:
        await update.message.reply_text(f"اتصال Binance (LIVE) غير مهيأ. جاري إعادة التهيئة...")
        client = initialize_binance_client(context)
        if not client:
            await update.message.reply_text(f"فشل إعادة تهيئة اتصال Binance (LIVE). تحقق من مفاتيح API.")
            return

    try:
        await update.message.reply_text(f"⏳ جاري جلب رصيد USDT (LIVE)...")
        client.ping()
        free_balance = get_usdt_balance(client)
        free_balance_str = format_decimal(free_balance, 2) # Show USDT balance with 2 decimals
        await update.message.reply_text(f"رصيد USDT المتاح (LIVE): {free_balance_str}")
        logger.info(f"تم عرض رصيد USDT للمستخدم {update.effective_user.id}: {free_balance_str}")
    except BinanceAPIException as e:
        logger.error(f"خطأ API من Binance في جلب رصيد USDT (LIVE): {e}")
        await update.message.reply_text(f"حدث خطأ أثناء جلب رصيد USDT من Binance (LIVE): {e}")
    except Exception as e:
        logger.error(f"خطأ في جلب رصيد USDT (LIVE): {e}")
        await update.message.reply_text(f"حدث خطأ غير متوقع أثناء جلب رصيد USDT (LIVE): {e}")

# --- Modified open_trades_command --- START
async def open_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    client = context.bot_data.get("binance_client")
    binance_mode = context.bot_data.get("binance_mode", "testnet")

    # --- Handle Testnet Mode --- 
    if binance_mode == "testnet":
        # Ensure trades are loaded
        if "open_trades_data" not in context.bot_data:
            load_testnet_trades(context.bot_data)

        open_trades_data = context.bot_data.get("open_trades_data", {})
        if not open_trades_data:
            await update.message.reply_text("لا توجد صفقات محاكاة مفتوحة في وضع Testnet.")
            return

        trades_text = "📊 **الصفقات المفتوحة (Testnet):**\n\n"
        total_simulated_value_usdt = Decimal("0")
        live_client = None # Need live client for current prices

        # Try to initialize a temporary live client for prices
        live_api_key = config["BINANCE"].get("LIVE_API_KEY")
        live_secret_key = config["BINANCE"].get("LIVE_SECRET_KEY")
        if live_api_key and live_api_key != "YOUR_BINANCE_LIVE_API_KEY_HERE" and \
           live_secret_key and live_secret_key != "YOUR_BINANCE_LIVE_SECRET_KEY_HERE":
            try:
                live_client = Client(live_api_key, live_secret_key, tld="com", testnet=False)
                live_client.ping()
            except Exception as e:
                logger.warning(f"فشل تهيئة عميل Live لجلب أسعار Testnet: {e}")
                live_client = None
        else:
            logger.warning("مفاتيح Live API غير متوفرة لجلب أسعار Testnet.")

        prices = {}
        if live_client:
            try:
                all_tickers = live_client.get_all_tickers()
                prices = {ticker['symbol']: Decimal(ticker['price']) for ticker in all_tickers}
            except Exception as e:
                logger.warning(f"فشل جلب أسعار السوق الحالية لتقييم صفقات Testnet: {e}")
                await update.message.reply_text("⚠️ لم نتمكن من جلب أسعار السوق الحالية للتقييم.")

        for symbol, trade_info in open_trades_data.items():
            quantity = trade_info.get("quantity", Decimal("0"))
            avg_price = trade_info.get("avg_price", Decimal("0"))
            cost_usdt = trade_info.get("cost_usdt", quantity * avg_price) # Recalculate if missing
            buy_timestamp = trade_info.get("buy_time")
            buy_time_str = datetime.fromtimestamp(buy_timestamp).strftime('%Y-%m-%d %H:%M') if buy_timestamp else "N/A"

            # Get precision (use live client if available, otherwise default)
            try:
                # Use live client's cache if available, otherwise testnet client's cache
                temp_context = ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=update.effective_chat.id, user_id=update.effective_user.id)
                # Share bot_data
                # We need a client instance to pass, but filters are in bot_data
                qty_precision, price_precision, _, _ = get_symbol_filters(temp_context, symbol)
            except ValueError:
                 qty_precision, price_precision = 8, 8 # Fallback

            trades_text += f"🔹 **{symbol}**\n"
            trades_text += f"   - الكمية: {format_decimal(quantity, qty_precision)}\n"
            trades_text += f"   - سعر الشراء: {format_decimal(avg_price, price_precision)}\n"
            trades_text += f"   - التكلفة: {format_decimal(cost_usdt, 2)} USDT\n"
            trades_text += f"   - وقت الشراء: {buy_time_str}\n"

            # Estimate current value and P/L
            current_price = prices.get(symbol)
            if current_price:
                current_value_usdt = quantity * current_price
                pnl_usdt = current_value_usdt - cost_usdt
                pnl_percent = (pnl_usdt / cost_usdt * 100) if cost_usdt > 0 else Decimal("0")
                trades_text += f"   - القيمة الحالية: **~{format_decimal(current_value_usdt, 2)} USDT**\n"
                pnl_sign = "➕" if pnl_usdt >= 0 else "➖"
                trades_text += f"   - الربح/الخسارة: {pnl_sign} {format_decimal(abs(pnl_usdt), 2)} USDT ({format_decimal(pnl_percent, 2)}%)\n"
                total_simulated_value_usdt += current_value_usdt
            else:
                trades_text += f"   - القيمة الحالية: (لا يتوفر سعر حالي)\n"
                total_simulated_value_usdt += cost_usdt # Add cost if no current price

            trades_text += "\n"

        trades_text += f"💰 **إجمالي قيمة الصفقات المقدرة:** **~{format_decimal(total_simulated_value_usdt, 2)} USDT**"

        # Split long messages if necessary (same logic as live mode)
        max_length = 4096
        if len(trades_text) > max_length:
            parts = []
            current_part = ""
            lines = trades_text.split('\n')
            for i, line in enumerate(lines):
                 if i == 0 and not current_part:
                     current_part = line + '\n'
                     continue
                 if len(current_part) + len(line) + 1 > max_length:
                     parts.append(current_part.strip())
                     current_part = line + '\n'
                 else:
                     current_part += line + '\n'
            parts.append(current_part.strip())
            for i, part in enumerate(parts):
                 if not part: continue
                 if i > 0: time.sleep(0.5)
                 await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(trades_text, parse_mode=ParseMode.MARKDOWN)

        return # End execution for testnet mode

    # --- Handle Live Mode --- (Code from previous step - unchanged)
    if binance_mode != "live":
         await update.message.reply_text(f"الأمر `/open_trades` لعرض الأرصدة الحقيقية يعمل فقط في وضع 'live'. الوضع الحالي: {binance_mode.upper()}")
         return

    if not client:
        await update.message.reply_text(f"اتصال Binance (LIVE) غير مهيأ. جاري إعادة التهيئة...")
        client = initialize_binance_client(context)
        if not client:
            await update.message.reply_text(f"فشل إعادة تهيئة اتصال Binance (LIVE). تحقق من مفاتيح API.")
            return

    try:
        await update.message.reply_text(f"⏳ جاري جلب أرصدة الحساب (LIVE)...")
        client.ping() # Check connection

        account_info = client.get_account()
        balances = account_info.get('balances', [])
        stablecoins = ["USDT", "BUSD", "USDC", "DAI", "TUSD", "PAX", "USDP"] # Add more if needed
        usdt_details = None
        portfolio_text = "📊 **أرصدة الحساب (LIVE):**\n\n"
        total_portfolio_value_usdt = Decimal("0")
        assets_found = False

        # Fetch current prices for valuation
        prices = {}
        try:
            all_tickers = client.get_all_tickers()
            prices = {ticker['symbol']: Decimal(ticker['price']) for ticker in all_tickers}
        except Exception as e:
            logger.warning(f"فشل جلب أسعار السوق الحالية للتقييم: {e}")
            await update.message.reply_text("⚠️ لم نتمكن من جلب أسعار السوق الحالية للتقييم.")
            # Continue without prices if fetching failed

        for balance in balances:
            asset = balance['asset']
            free = Decimal(balance['free'])
            locked = Decimal(balance['locked'])
            total_balance = free + locked

            if total_balance > 0:
                # Store USDT details separately
                if asset == "USDT":
                    usdt_details = balance
                    total_portfolio_value_usdt += total_balance # Add USDT balance directly
                    continue # Skip displaying USDT in the main list

                # Keep track of other stablecoins for total value but don't display details
                is_stable = asset in stablecoins
                if is_stable:
                     total_portfolio_value_usdt += total_balance # Add other stablecoins directly
                     continue # Skip displaying stablecoins other than USDT

                # Display non-zero, non-stablecoin assets
                assets_found = True
                portfolio_text += f"🔹 **{asset}**\n"
                portfolio_text += f"   - متاح: {format_decimal(free, 8)}\n"
                portfolio_text += f"   - مقفل: {format_decimal(locked, 8)}\n"
                portfolio_text += f"   - الإجمالي: {format_decimal(total_balance, 8)}\n"

                # Try to find price and estimate value
                pair = f"{asset}USDT"
                current_price = prices.get(pair)
                if current_price:
                    value_usdt = total_balance * current_price
                    portfolio_text += f"   - القيمة المقدرة: **~{format_decimal(value_usdt, 2)} USDT**\n"
                    total_portfolio_value_usdt += value_usdt
                else:
                    portfolio_text += f"   - القيمة المقدرة: (لا يتوفر سعر لـ {pair})\n"

                portfolio_text += "\n" # Add space between assets

        # Display USDT details at the end
        if usdt_details:
            assets_found = True # Mark as found even if only USDT exists
            portfolio_text += f"💲 **USDT**\n"
            portfolio_text += f"   - متاح: {format_decimal(usdt_details['free'], 2)}\n" # Show USDT with 2 decimals
            portfolio_text += f"   - مقفل: {format_decimal(usdt_details['locked'], 2)}\n"
            portfolio_text += f"   - الإجمالي: {format_decimal(Decimal(usdt_details['free']) + Decimal(usdt_details['locked']), 2)}\n\n"

        if not assets_found and not usdt_details: # Check if anything was displayed
             portfolio_text = "لم يتم العثور على أرصدة ذات قيمة لعرضها (بخلاف العملات المستقرة الأخرى)."
        else:
            portfolio_text += f"💰 **إجمالي قيمة المحفظة المقدرة:** **~{format_decimal(total_portfolio_value_usdt, 2)} USDT**"

        # Split long messages if necessary
        max_length = 4096
        if len(portfolio_text) > max_length:
            parts = []
            current_part = "" # Start empty
            lines = portfolio_text.split('\n')
            for i, line in enumerate(lines):
                 # Add header to the first part
                 if i == 0 and not current_part:
                     current_part = line + '\n'
                     continue
                 # Check if adding the next line exceeds max length
                 if len(current_part) + len(line) + 1 > max_length:
                     parts.append(current_part.strip()) # Add the completed part
                     current_part = line + '\n' # Start a new part with the current line
                 else:
                     current_part += line + '\n'
            parts.append(current_part.strip()) # Add the last part

            for i, part in enumerate(parts):
                 if not part: continue # Skip empty parts
                 if i > 0: time.sleep(0.5) # Small delay between messages
                 await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(portfolio_text, parse_mode=ParseMode.MARKDOWN)

    except BinanceAPIException as e:
        logger.error(f"خطأ API من Binance في جلب الأرصدة (LIVE): {e}")
        await update.message.reply_text(f"حدث خطأ أثناء جلب البيانات من Binance (LIVE): {e}")
    except Exception as e:
        logger.exception(f"خطأ في جلب الأرصدة (LIVE): {e}") # Use exception for full traceback
        await update.message.reply_text(f"حدث خطأ غير متوقع (LIVE): {e}")
# --- Modified open_trades_command --- END

async def manual_mode_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["trading_mode"] = "manual"
    await update.message.reply_text("تم تفعيل وضع التداول اليدوي.")
    logger.info(f"المستخدم {update.effective_user.id} قام بتفعيل الوضع اليدوي.")

async def manual_mode_off_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.bot_data.get("trading_mode") == "manual":
        context.bot_data["trading_mode"] = "off"
        await update.message.reply_text("تم إيقاف وضع التداول اليدوي.")
        logger.info(f"المستخدم {update.effective_user.id} قام بإيقاف الوضع اليدوي.")
    else:
        await update.message.reply_text("وضع التداول اليدوي غير مفعل حالياً.")

async def auto_mode_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Parse optional arguments for amount and profit %
    try:
        if len(context.args) >= 1:
            new_amount = Decimal(context.args[0])
            if new_amount <= 0:
                raise ValueError("المبلغ يجب أن يكون أكبر من صفر.")
            context.bot_data["auto_trade_amount_usdt"] = new_amount
            logger.info(f"تحديث مبلغ التداول الآلي إلى: {new_amount}")
        if len(context.args) >= 2:
            new_profit = Decimal(context.args[1])
            if new_profit <= 0:
                raise ValueError("نسبة الربح يجب أن تكون أكبر من صفر.")
            context.bot_data["auto_trade_profit_percent"] = new_profit
            logger.info(f"تحديث نسبة الربح الآلية إلى: {new_profit}%")

    except (ValueError, InvalidOperation) as e:
        await update.message.reply_text(f"خطأ في المدخلات: {e}. يرجى إدخال أرقام صالحة للمبلغ ونسبة الربح.")
        return

    context.bot_data["trading_mode"] = "auto"
    current_amount = context.bot_data["auto_trade_amount_usdt"]
    current_profit = context.bot_data["auto_trade_profit_percent"]
    await update.message.reply_text(
        f"تم تفعيل وضع التداول الآلي.\n"
        f"مبلغ الصفقة: {format_decimal(current_amount, 2)} USDT\n"
        f"نسبة الربح المستهدفة: {format_decimal(current_profit, 2)}%"
    )
    logger.info(f"المستخدم {update.effective_user.id} قام بتفعيل الوضع الآلي (المبلغ: {current_amount}, الربح: {current_profit}%)." )

async def auto_mode_off_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.bot_data.get("trading_mode") == "auto":
        context.bot_data["trading_mode"] = "off"
        await update.message.reply_text("تم إيقاف وضع التداول الآلي.")
        logger.info(f"المستخدم {update.effective_user.id} قام بإيقاف الوضع الآلي.")
    else:
        await update.message.reply_text("وضع التداول الآلي غير مفعل حالياً.")

async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Check if manual mode is on
    if context.bot_data.get("trading_mode") != "manual":
        await update.message.reply_text("الشراء اليدوي يتطلب تفعيل الوضع اليدوي أولاً باستخدام /manual_mode_on")
        return

    # Check if trading is active
    if not context.bot_data.get("trading_active", True):
        await update.message.reply_text("التداول متوقف مؤقتاً. استخدم /start_trading للاستئناف.")
        return

    # Parse arguments: /buy SYMBOL AMOUNT_USDT [PROFIT%]
    if len(context.args) < 2:
        await update.message.reply_text("الاستخدام: /buy <الرمز> <مبلغ_USDT> [نسبة_الربح%]")
        return

    symbol = context.args[0].upper()
    try:
        amount_usdt = Decimal(context.args[1])
        if amount_usdt <= 0:
            raise ValueError("مبلغ USDT يجب أن يكون أكبر من صفر.")
        profit_percent = None
        if len(context.args) >= 3:
            profit_percent = Decimal(context.args[2])
            if profit_percent <= 0:
                raise ValueError("نسبة الربح يجب أن تكون أكبر من صفر.")
    except (ValueError, InvalidOperation) as e:
        await update.message.reply_text(f"خطأ في المدخلات: {e}. يرجى إدخال أرقام صالحة.")
        return

    client = context.bot_data.get("binance_client")
    binance_mode = context.bot_data.get("binance_mode", "testnet")

    if not client:
        await update.message.reply_text(f"اتصال Binance ({binance_mode.upper()}) غير مهيأ. لا يمكن تنفيذ الشراء.")
        return

    # Validate symbol
    if not validate_symbol(client, context, symbol):
        await update.message.reply_text(f"رمز العملة {symbol} غير صالح أو غير موجود في Binance.")
        return

    try:
        # Get precision and filters
        qty_precision, price_precision, min_qty, step_size = get_symbol_filters(context, symbol)

        # Get current price
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = Decimal(ticker['price'])
        if current_price <= 0:
             raise ValueError(f"سعر السوق الحالي لـ {symbol} غير صالح ({current_price}).")

        # Calculate initial quantity
        quantity = amount_usdt / current_price

        # Adjust quantity for LOT_SIZE filter
        adjusted_quantity = adjust_quantity_for_lot_size(quantity, min_qty, step_size)
        if adjusted_quantity is None or adjusted_quantity <= 0:
            # Error message already logged by adjust_quantity_for_lot_size
            await update.message.reply_text(f"❌ فشل حساب كمية صالحة للشراء لـ {symbol} بالمبلغ {amount_usdt} USDT. قد يكون المبلغ صغيراً جداً أو قواعد الزوج غير متوافقة.")
            return

        formatted_quantity = format_decimal(adjusted_quantity, qty_precision)
        logger.info(f"شراء يدوي لـ {symbol}: المبلغ={amount_usdt} USDT, السعر={current_price}, الكمية الأولية={quantity}, الكمية المعدلة={adjusted_quantity}, الكمية المنسقة={formatted_quantity}")

        # Check balance
        if binance_mode == "live":
            usdt_balance = get_usdt_balance(client)
            if usdt_balance < amount_usdt:
                await update.message.reply_text(f"❌ رصيد USDT غير كافٍ ({format_decimal(usdt_balance, 2)}). مطلوب: {format_decimal(amount_usdt, 2)} USDT.")
                return
        else: # Testnet balance check
            if "simulated_usdt_balance" not in context.bot_data: load_testnet_trades(context.bot_data)
            simulated_usdt_balance = context.bot_data.get("simulated_usdt_balance", Decimal("0"))
            if simulated_usdt_balance < amount_usdt:
                await update.message.reply_text(f"❌ رصيد USDT المحاكى غير كافٍ ({format_decimal(simulated_usdt_balance, 2)}). مطلوب: {format_decimal(amount_usdt, 2)} USDT.")
                return

        # Execute market buy order
        await update.message.reply_text(f"⏳ جاري تنفيذ أمر شراء سوق لـ {formatted_quantity} {symbol} بقيمة ~{amount_usdt} USDT...")
        order = client.order_market_buy(symbol=symbol, quantity=formatted_quantity)
        logger.info(f"تم تنفيذ أمر الشراء اليدوي: {order}")

        # Process fills to get actual buy price and quantity
        filled_qty = Decimal(order.get('executedQty', '0'))
        cummulative_quote_qty = Decimal(order.get('cummulativeQuoteQty', '0'))
        avg_buy_price = Decimal("0")
        if filled_qty > 0:
            avg_buy_price = cummulative_quote_qty / filled_qty
        else:
             # If filled quantity is 0, something went wrong despite API success
             logger.warning(f"أمر الشراء لـ {symbol} تم بنجاح ولكن الكمية المنفذة صفر؟ {order}")
             await update.message.reply_text(f"⚠️ أمر الشراء تم ولكن لم يتم تنفيذ أي كمية. تحقق من Binance.")
             return

        buy_confirmation = (
            f"✅ تم شراء {format_decimal(filled_qty, qty_precision)} {symbol} بنجاح!\n"
            f"   متوسط سعر الشراء: {format_decimal(avg_buy_price, price_precision)}\n"
            f"   التكلفة الإجمالية: {format_decimal(cummulative_quote_qty, 2)} USDT"
        )
        await update.message.reply_text(buy_confirmation)

        # --- Testnet Trade Saving Logic --- START
        if binance_mode == "testnet":
            # Ensure trades dict exists
            context.bot_data.setdefault("open_trades_data", {})
            # Check if trade for this symbol already exists
            if symbol in context.bot_data["open_trades_data"]:
                # Update existing trade (average down)
                existing_trade = context.bot_data["open_trades_data"][symbol]
                old_qty = existing_trade["quantity"]
                old_cost = existing_trade["cost_usdt"]
                new_total_qty = old_qty + filled_qty
                new_total_cost = old_cost + cummulative_quote_qty
                new_avg_price = new_total_cost / new_total_qty if new_total_qty > 0 else Decimal("0")
                trade_data = {
                    "quantity": new_total_qty,
                    "avg_price": new_avg_price,
                    "buy_time": time.time(), # Update time to latest buy
                    "cost_usdt": new_total_cost
                }
                logger.info(f"تحديث صفقة Testnet لـ {symbol}: الكمية الجديدة={new_total_qty}, المتوسط الجديد={new_avg_price}")
            else:
                # Add new trade
                trade_data = {
                    "quantity": filled_qty,
                    "avg_price": avg_buy_price,
                    "buy_time": time.time(),
                    "cost_usdt": cummulative_quote_qty
                }
                logger.info(f"إضافة صفقة Testnet جديدة لـ {symbol}")

            context.bot_data["open_trades_data"][symbol] = trade_data
            # Deduct from simulated balance
            context.bot_data["simulated_usdt_balance"] -= cummulative_quote_qty
            save_testnet_trades(context.bot_data) # Save after modification
        # --- Testnet Trade Saving Logic --- END

        # Place Take Profit order if profit % is specified (Live mode only)
        if profit_percent is not None and binance_mode == "live":
            await update.message.reply_text(f"⏳ جاري وضع أمر جني أرباح عند {profit_percent}%...")
            tp_order = await place_take_profit_order(client, context, symbol, filled_qty, avg_buy_price, profit_percent)
            if tp_order:
                await update.message.reply_text(f"✅ تم وضع أمر جني الأرباح بنجاح (ID: {tp_order.get('orderId')}).")
            else:
                await update.message.reply_text("⚠️ فشل وضع أمر جني الأرباح.")

    except (BinanceAPIException, BinanceOrderException) as e:
        logger.error(f"خطأ Binance API/Order أثناء الشراء اليدوي لـ {symbol}: {e}")
        await update.message.reply_text(f"❌ فشل أمر الشراء: {e}")
    except ValueError as e:
        logger.error(f"خطأ في القيمة أثناء الشراء اليدوي لـ {symbol}: {e}")
        await update.message.reply_text(f"❌ خطأ: {e}")
    except Exception as e:
        logger.exception(f"خطأ غير متوقع أثناء الشراء اليدوي لـ {symbol}: {e}")
        await update.message.reply_text(f"❌ حدث خطأ غير متوقع أثناء محاولة الشراء.")

# --- Sell Command (Testnet Only) --- START
async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    binance_mode = context.bot_data.get("binance_mode", "testnet")
    if binance_mode != "testnet":
        await update.message.reply_text("أمر البيع المحاكى `/sell` يعمل فقط في وضع Testnet.")
        return

    # Parse arguments: /sell SYMBOL [QUANTITY]
    if not context.args or len(context.args) < 1:
        await update.message.reply_text("الاستخدام: /sell <الرمز> [الكمية] (بيع الكل إذا لم تحدد كمية)")
        return

    symbol = context.args[0].upper()
    sell_quantity_str = context.args[1] if len(context.args) > 1 else None

    # Ensure trades are loaded
    if "open_trades_data" not in context.bot_data:
        load_testnet_trades(context.bot_data)
    open_trades_data = context.bot_data.get("open_trades_data", {})

    # Check if trade exists
    if symbol not in open_trades_data:
        await update.message.reply_text(f"لا توجد صفقة مفتوحة للرمز {symbol} في وضع Testnet.")
        return

    existing_trade = open_trades_data[symbol]
    available_quantity = existing_trade["quantity"]

    # Get filters for precision and validation
    try:
        qty_precision, price_precision, min_qty, step_size = get_symbol_filters(context, symbol)
    except ValueError as e:
        await update.message.reply_text(f"❌ خطأ في جلب معلومات الرمز {symbol}: {e}")
        return

    # Determine quantity to sell
    quantity_to_sell = available_quantity # Default to selling all
    if sell_quantity_str:
        try:
            quantity_to_sell = Decimal(sell_quantity_str)
            if quantity_to_sell <= 0:
                raise ValueError("كمية البيع يجب أن تكون أكبر من صفر.")
            # Check against available quantity *before* adjustment
            if quantity_to_sell > available_quantity:
                await update.message.reply_text(f"❌ لا يمكنك بيع {format_decimal(quantity_to_sell, qty_precision)} {symbol}. الكمية المتاحة: {format_decimal(available_quantity, qty_precision)}")
                return
        except (ValueError, InvalidOperation) as e:
            await update.message.reply_text(f"خطأ في كمية البيع: {e}. يرجى إدخال رقم صالح أو تركها فارغة لبيع الكل.")
            return

    # Adjust the quantity to sell based on LOT_SIZE rules
    adjusted_quantity_to_sell = adjust_quantity_for_lot_size(quantity_to_sell, min_qty, step_size)
    if adjusted_quantity_to_sell is None or adjusted_quantity_to_sell <= 0:
        await update.message.reply_text(f"❌ فشل تعديل كمية البيع ({quantity_to_sell}) لتتوافق مع قواعد {symbol}. قد تكون الكمية صغيرة جداً.")
        return

    # Final check: ensure adjusted quantity doesn't exceed available
    if adjusted_quantity_to_sell > available_quantity:
         logger.error(f"خطأ منطقي: الكمية المعدلة للبيع {adjusted_quantity_to_sell} أكبر من المتاحة {available_quantity} لـ {symbol}")
         await update.message.reply_text(f"❌ خطأ داخلي: الكمية المعدلة للبيع أكبر من المتاحة. يرجى المحاولة مرة أخرى أو بيع الكل.")
         return

    # Get Live client for current price
    live_client = None
    live_api_key = config["BINANCE"].get("LIVE_API_KEY")
    live_secret_key = config["BINANCE"].get("LIVE_SECRET_KEY")
    if live_api_key and live_api_key != "YOUR_BINANCE_LIVE_API_KEY_HERE" and \
       live_secret_key and live_secret_key != "YOUR_BINANCE_LIVE_SECRET_KEY_HERE":
        try:
            live_client = Client(live_api_key, live_secret_key, tld="com", testnet=False)
            live_client.ping()
        except Exception as e:
            logger.warning(f"فشل تهيئة عميل Live لجلب سعر البيع المحاكى: {e}")
            live_client = None
    else:
        logger.warning("مفاتيح Live API غير متوفرة لجلب سعر البيع المحاكى.")

    if not live_client:
        await update.message.reply_text("⚠️ لا يمكن جلب سعر السوق الحالي للمحاكاة. فشل تهيئة اتصال Live API.")
        return

    try:
        # Get current price from Live API
        ticker = live_client.get_symbol_ticker(symbol=symbol)
        current_price = Decimal(ticker['price'])

        # Use the adjusted and formatted quantity
        formatted_quantity_to_sell = format_decimal(adjusted_quantity_to_sell, qty_precision)
        # Convert back to Decimal for calculation
        decimal_formatted_quantity_to_sell = Decimal(formatted_quantity_to_sell)

        # Calculate simulated sell value
        simulated_sell_value_usdt = decimal_formatted_quantity_to_sell * current_price

        await update.message.reply_text(f"⏳ جاري محاكاة بيع {formatted_quantity_to_sell} {symbol} بسعر السوق ~{format_decimal(current_price, price_precision)}...")

        # Update simulated balance
        context.bot_data["simulated_usdt_balance"] += simulated_sell_value_usdt

        # Update or remove trade data
        remaining_quantity = available_quantity - decimal_formatted_quantity_to_sell
        # Use a small tolerance for floating point comparisons
        if remaining_quantity <= Decimal(f"1e-{qty_precision}"):
            del context.bot_data["open_trades_data"][symbol]
            logger.info(f"تمت إزالة صفقة Testnet لـ {symbol} بعد بيع الكل.")
            sell_type = "بيع الكل"
        else:
            # Update existing trade (partial sell)
            original_cost = existing_trade["cost_usdt"]
            cost_per_unit = original_cost / available_quantity if available_quantity > 0 else Decimal("0")
            new_cost = remaining_quantity * cost_per_unit

            context.bot_data["open_trades_data"][symbol]["quantity"] = remaining_quantity
            context.bot_data["open_trades_data"][symbol]["cost_usdt"] = new_cost # Update cost proportionally
            logger.info(f"تحديث صفقة Testnet لـ {symbol} بعد بيع جزئي. الكمية المتبقية: {remaining_quantity}")
            sell_type = "بيع جزئي"

        # Save changes
        save_testnet_trades(context.bot_data)

        # Send confirmation
        sell_confirmation = (
            f"✅ تم بيع {formatted_quantity_to_sell} {symbol} (محاكاة) بنجاح!\n"
            f"   سعر البيع المحاكى: {format_decimal(current_price, price_precision)}\n"
            f"   القيمة المحصلة: +{format_decimal(simulated_sell_value_usdt, 2)} USDT\n"
            f"   نوع البيع: {sell_type}\n"
            f"   رصيد USDT المحاكى الجديد: {format_decimal(context.bot_data['simulated_usdt_balance'], 2)} USDT"
        )
        await update.message.reply_text(sell_confirmation)

    except BinanceAPIException as e:
        logger.error(f"خطأ Binance API أثناء محاكاة البيع لـ {symbol}: {e}")
        await update.message.reply_text(f"❌ فشل جلب سعر السوق للمحاكاة: {e}")
    except ValueError as e:
        logger.error(f"خطأ في القيمة أثناء محاكاة البيع لـ {symbol}: {e}")
        await update.message.reply_text(f"❌ خطأ: {e}")
    except Exception as e:
        logger.exception(f"خطأ غير متوقع أثناء محاكاة البيع لـ {symbol}: {e}")
        await update.message.reply_text(f"❌ حدث خطأ غير متوقع أثناء محاولة البيع.")
# --- Sell Command (Testnet Only) --- END

async def stop_trading_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["trading_active"] = False
    await update.message.reply_text("تم إيقاف التداول مؤقتًا. لن يتم فتح صفقات جديدة.")
    logger.info(f"المستخدم {update.effective_user.id} أوقف التداول.")

async def start_trading_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["trading_active"] = True
    await update.message.reply_text("تم استئناف التداول. يمكن فتح صفقات جديدة.")
    logger.info(f"المستخدم {update.effective_user.id} استأنف التداول.")

async def profit_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        closed_trades = context.bot_data.get("closed_trades", [])
        if not closed_trades:
            await update.message.reply_text("📭 لا توجد صفقات مغلقة بعد.")
            return

        total_profit = Decimal("0")
        lines = []
        for trade in closed_trades:
            symbol = trade.get("symbol", "N/A")
            entry = Decimal(trade.get("entry_price", "0"))
            exit_p = Decimal(trade.get("exit_price", "0"))
            qty = Decimal(trade.get("quantity", "0"))
            profit = Decimal(trade.get("profit", "0"))
            total_profit += profit
            emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            lines.append(f"{emoji} {symbol}: {profit:.2f} USDT")


        summary = "\n".join(lines)
        await update.message.reply_text(
            f"📊 ملخص الأرباح المغلقة:\n{summary}\n\n💰 الإجمالي: {total_profit:.2f} USDT",
            parse_mode=ParseMode.MARKDOWN
        )

    except Exception as e:
        await update.message.reply_text(f"❌ خطأ في عرض ملخص الأرباح: {e}")

async def analyze_custom_signal(update, context):
    message = update.message.text
    if not message:
        return

    # --- استخراج المعلومات من التوصية ---
    pair_match = re.search(r"Pair:\s*([A-Z]+/[A-Z]+)", message)
    entry_match = re.search(r"Entry Price:\s*([0-9.]+)", message)
    targets_match = re.findall(r"Target \d+:\s*([0-9.]+)", message)

    if not pair_match or not entry_match:
        return

    # --- تحويل البيانات ---
    symbol_raw = pair_match.group(1).replace("/", "")
    entry_price = Decimal(entry_match.group(1))
    targets = [Decimal(p) for p in targets_match]

    if not targets:
        return

    client = context.bot_data.get("binance_client")
    if not client:
        return

    ticker = client.get_symbol_ticker(symbol=symbol_raw)
    current_price = Decimal(ticker['price'])

    if current_price <= entry_price:
        amount_usdt = context.bot_data.get("auto_trade_amount_usdt", Decimal("50"))
        profit_percent = context.bot_data.get("auto_trade_profit_percent", Decimal("5"))

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await update.message.reply_text(
            f"✅ تم تفعيل التوصية لـ {symbol_raw}\n"
            f"📉 السعر الحالي: {current_price}\n"
            f"🎯 سعر الدخول: {entry_price}\n"
            f"⏱️ وقت التفعيل: {now_str}"
        )

        try:
            qty_precision, price_precision, min_qty, step_size = get_symbol_filters(context, symbol_raw)
            quantity = amount_usdt / current_price
            adjusted_quantity = adjust_quantity_for_lot_size(quantity, min_qty, step_size)
            if adjusted_quantity is None or adjusted_quantity <= 0:
                await update.message.reply_text("❌ فشل في حساب كمية صالحة للتداول.")
                return

            formatted_quantity = format_decimal(adjusted_quantity, qty_precision)
            mode = context.bot_data.get("binance_mode", "testnet")

            if mode == "live":
                order = client.order_market_buy(symbol=symbol_raw, quantity=formatted_quantity)
                filled_qty = Decimal(order.get('executedQty', '0'))
                quote_qty = Decimal(order.get('cummulativeQuoteQty', '0'))
                avg_price = quote_qty / filled_qty if filled_qty > 0 else current_price

                await update.message.reply_text(
                    f"✅ تم الشراء: {formatted_quantity} {symbol_raw} بسعر {format_decimal(avg_price, price_precision)}"
                )

                tp_price = avg_price * (Decimal("1") + profit_percent / Decimal("100"))
                formatted_tp_price = format_decimal(tp_price, price_precision)

                tp_order = client.order_limit_sell(
                    symbol=symbol_raw,
                    quantity=format_decimal(filled_qty, qty_precision),
                    price=formatted_tp_price
                )

                await update.message.reply_text(
                    f"📈 تم وضع أمر بيع بهدف الربح عند {formatted_tp_price}"
                )

            else:
                if "simulated_usdt_balance" not in context.bot_data:
                    context.bot_data["simulated_usdt_balance"] = Decimal("10000")

                if context.bot_data["simulated_usdt_balance"] < amount_usdt:
                    await update.message.reply_text("❌ رصيد USDT غير كافٍ في Testnet.")
                    return

                avg_price = current_price
                cost = adjusted_quantity * avg_price
                trade_data = {
                    "quantity": adjusted_quantity,
                    "avg_price": avg_price,
                    "buy_time": time.time(),
                    "cost_usdt": cost
                }
                context.bot_data.setdefault("open_trades_data", {})[symbol_raw] = trade_data
                context.bot_data["simulated_usdt_balance"] -= cost

                await update.message.reply_text(
                    f"🧪 تم شراء محاكاة: {formatted_quantity} {symbol_raw} بسعر {format_decimal(avg_price, price_precision)}"
                )

                save_testnet_trades(context.bot_data)

        except (BinanceAPIException, BinanceOrderException) as e:
            await update.message.reply_text(f"❌ خطأ من Binance: {e}")
        except Exception as e:
            await update.message.reply_text(f"❌ خطأ غير متوقع: {e}")

    else:
        await update.message.reply_text(
            f"📛 لم يتم تفعيل التوصية لـ {symbol_raw} لأن السعر الحالي ({current_price}) أعلى من سعر الدخول ({entry_price})."
        )

# --- Signal Processing --- 
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await analyze_custom_signal(update, context)

    # Check if auto mode is on
    if context.bot_data.get("trading_mode") != "auto":
        # logger.debug("تجاهل الرسالة، الوضع الآلي غير مفعل.")
        return

    # Check if trading is active
    if not context.bot_data.get("trading_active", True):
        logger.info("تجاهل الإشارة، التداول متوقف مؤقتًا.")
        # Optionally notify user?
        # await update.message.reply_text("تلقيت إشارة ولكن التداول متوقف مؤقتًا.")
        return

    message_text = update.message.text
    if not message_text:
        return

    # Basic signal parsing (Example: looking for BUY SYMBOL)
    # This needs to be adapted to the actual signal format
    match = re.search(r"BUY\s+([A-Z]+USDT)", message_text, re.IGNORECASE)
    if not match:
        # logger.debug(f"الرسالة لا تطابق نمط إشارة الشراء: {message_text[:50]}...")
        return

    symbol = match.group(1).upper()
    logger.info(f"تم اكتشاف إشارة شراء محتملة لـ {symbol}")

    client = context.bot_data.get("binance_client")
    binance_mode = context.bot_data.get("binance_mode", "testnet")

    if not client:
        logger.error(f"لا يمكن معالجة إشارة {symbol}، عميل Binance ({binance_mode.upper()}) غير مهيأ.")
        # Optionally notify user?
        # await update.message.reply_text(f"خطأ: لا يمكن معالجة إشارة {symbol} بسبب مشكلة في الاتصال بـ Binance.")
        return

    # Validate symbol
    if not validate_symbol(client, context, symbol):
        logger.warning(f"تجاهل الإشارة، الرمز {symbol} غير صالح.")
        return

    try:
        amount_usdt = context.bot_data["auto_trade_amount_usdt"]
        profit_percent = context.bot_data["auto_trade_profit_percent"]

        # Get precision and filters
        qty_precision, price_precision, min_qty, step_size = get_symbol_filters(context, symbol)

        # Get current price
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = Decimal(ticker['price'])
        if current_price <= 0:
             raise ValueError(f"سعر السوق الحالي لـ {symbol} غير صالح ({current_price}).")

        # Calculate initial quantity
        quantity = amount_usdt / current_price

        # Adjust quantity for LOT_SIZE filter
        adjusted_quantity = adjust_quantity_for_lot_size(quantity, min_qty, step_size)
        if adjusted_quantity is None or adjusted_quantity <= 0:
            logger.error(f"فشل حساب كمية صالحة للشراء الآلي لـ {symbol} بالمبلغ {amount_usdt} USDT.")
            # Optionally notify user?
            # await update.message.reply_text(f"❌ فشل حساب كمية صالحة للشراء الآلي لـ {symbol}.")
            return

        formatted_quantity = format_decimal(adjusted_quantity, qty_precision)
        logger.info(f"معالجة إشارة {symbol}: المبلغ={amount_usdt} USDT, السعر={current_price}, الكمية الأولية={quantity}, الكمية المعدلة={adjusted_quantity}, الكمية المنسقة={formatted_quantity}")

        # Check balance
        if binance_mode == "live":
            usdt_balance = get_usdt_balance(client)
            if usdt_balance < amount_usdt:
                logger.warning(f"رصيد USDT غير كافٍ ({usdt_balance}) للتداول الآلي لـ {symbol}. مطلوب: {amount_usdt}")
                context.bot_data["insufficient_balance_auto_mode"] = True
                # Optionally notify user?
                # await update.message.reply_text(f"⚠️ رصيد USDT غير كافٍ لتنفيذ إشارة {symbol}.")
                return
            else:
                context.bot_data.pop("insufficient_balance_auto_mode", None) # Clear flag if balance is sufficient
        else: # Testnet balance check
            if "simulated_usdt_balance" not in context.bot_data: load_testnet_trades(context.bot_data)
            simulated_usdt_balance = context.bot_data.get("simulated_usdt_balance", Decimal("0"))
            if simulated_usdt_balance < amount_usdt:
                logger.warning(f"رصيد USDT المحاكى غير كافٍ ({simulated_usdt_balance}) للتداول الآلي لـ {symbol}. مطلوب: {amount_usdt}")
                context.bot_data["insufficient_balance_auto_mode"] = True
                return
            else:
                context.bot_data.pop("insufficient_balance_auto_mode", None)

        # Execute market buy order
        await update.message.reply_text(f"🤖 تلقيت إشارة شراء لـ {symbol}. جاري تنفيذ أمر شراء سوق بقيمة ~{amount_usdt} USDT...")
        order = client.order_market_buy(symbol=symbol, quantity=formatted_quantity)
        logger.info(f"تم تنفيذ أمر الشراء الآلي بناءً على الإشارة: {order}")

        # Process fills
        filled_qty = Decimal(order.get('executedQty', '0'))
        cummulative_quote_qty = Decimal(order.get('cummulativeQuoteQty', '0'))
        avg_buy_price = Decimal("0")
        if filled_qty > 0:
            avg_buy_price = cummulative_quote_qty / filled_qty
        else:
             logger.warning(f"أمر الشراء الآلي لـ {symbol} تم بنجاح ولكن الكمية المنفذة صفر؟ {order}")
             await update.message.reply_text(f"⚠️ أمر الشراء الآلي تم ولكن لم يتم تنفيذ أي كمية. تحقق من Binance.")
             return

        buy_confirmation = (
            f"✅ تم شراء {format_decimal(filled_qty, qty_precision)} {symbol} (آلي) بنجاح!\n"
            f"   متوسط سعر الشراء: {format_decimal(avg_buy_price, price_precision)}\n"
            f"   التكلفة الإجمالية: {format_decimal(cummulative_quote_qty, 2)} USDT"
        )
        await update.message.reply_text(buy_confirmation)

        # --- Testnet Trade Saving Logic --- START
        if binance_mode == "testnet":
            context.bot_data.setdefault("open_trades_data", {})
            if symbol in context.bot_data["open_trades_data"]:
                existing_trade = context.bot_data["open_trades_data"][symbol]
                old_qty = existing_trade["quantity"]
                old_cost = existing_trade["cost_usdt"]
                new_total_qty = old_qty + filled_qty
                new_total_cost = old_cost + cummulative_quote_qty
                new_avg_price = new_total_cost / new_total_qty if new_total_qty > 0 else Decimal("0")
                trade_data = {
                    "quantity": new_total_qty,
                    "avg_price": new_avg_price,
                    "buy_time": time.time(),
                    "cost_usdt": new_total_cost
                }
                logger.info(f"تحديث صفقة Testnet لـ {symbol} (آلي): الكمية الجديدة={new_total_qty}, المتوسط الجديد={new_avg_price}")
            else:
                trade_data = {
                    "quantity": filled_qty,
                    "avg_price": avg_buy_price,
                    "buy_time": time.time(),
                    "cost_usdt": cummulative_quote_qty
                }
                logger.info(f"إضافة صفقة Testnet جديدة لـ {symbol} (آلي)")

            context.bot_data["open_trades_data"][symbol] = trade_data
            # Deduct from simulated balance
            context.bot_data["simulated_usdt_balance"] -= cummulative_quote_qty
            save_testnet_trades(context.bot_data)
        # --- Testnet Trade Saving Logic --- END

        # Place Take Profit order (Live mode only)
        if profit_percent is not None and binance_mode == "live":
            await update.message.reply_text(f"⏳ جاري وضع أمر جني أرباح عند {profit_percent}%...")
            tp_order = await place_take_profit_order(client, context, symbol, filled_qty, avg_buy_price, profit_percent)
            if tp_order:
                await update.message.reply_text(f"✅ تم وضع أمر جني الأرباح بنجاح (ID: {tp_order.get('orderId')}).")
            else:
                await update.message.reply_text("⚠️ فشل وضع أمر جني الأرباح.")

    except (BinanceAPIException, BinanceOrderException) as e:
        logger.error(f"خطأ Binance API/Order أثناء معالجة إشارة {symbol}: {e}")
        await update.message.reply_text(f"❌ فشل تنفيذ إشارة الشراء لـ {symbol}: {e}")
    except ValueError as e:
        logger.error(f"خطأ في القيمة أثناء معالجة إشارة {symbol}: {e}")
        await update.message.reply_text(f"❌ خطأ في معالجة إشارة {symbol}: {e}")
    except Exception as e:
        logger.exception(f"خطأ غير متوقع أثناء معالجة إشارة {symbol}: {e}")
        await update.message.reply_text(f"❌ حدث خطأ غير متوقع أثناء معالجة إشارة {symbol}.")

# --- Error Handler --- 
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"حدث استثناء أثناء معالجة تحديث: {context.error}", exc_info=context.error)
    # Optionally send a message to the user or a specific admin chat
    # update_str = update.to_json() if isinstance(update, Update) else str(update)
    # try:
    #     await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Bot Error: {context.error}\nUpdate: {update_str}")
    # except Exception as e_send:
    #     logger.error(f"Failed to send error message to admin: {e_send}")



async def place_stop_loss_order(client, symbol, quantity, stop_price, testnet=False):
    try:
        if testnet:
            order = client.create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=quantity
            )
            logger.info(f"🚨 تم تنفيذ وقف الخسارة فوراً (Testnet): {order}")
            return order
        else:
            order = client.create_order(
                symbol=symbol,
                side="SELL",
                type="STOP_MARKET",
                stopPrice=str(stop_price),
                quantity=quantity
            )
            logger.info(f"🚨 تم إنشاء أمر وقف خسارة: {order}")
            return order
    except Exception as e:
        logger.error(f"فشل إنشاء أمر وقف الخسارة: {e}")
        return None



async def open_trades_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        client = context.bot_data.get("binance_client")
        trades = context.bot_data.get("open_trades_data", {})
        if not trades:
            await update.message.reply_text("📭 لا توجد صفقات مفتوحة حالياً.")
            return

        message_lines = []
        for symbol, trade in trades.items():
            quantity = Decimal(trade.get("quantity", "0"))
            avg_price = Decimal(trade.get("avg_price", "0"))
            cost = Decimal(trade.get("cost_usdt", "0"))
            current = Decimal(client.get_symbol_ticker(symbol=symbol)["price"])
            value_now = current * quantity
            pnl = value_now - cost
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            message_lines.append(f"{emoji} {symbol}: {pnl:.2f} USDT | الكمية: {quantity}")

        await update.message.reply_text("📈 ملخص الصفقات المفتوحة:\n" + "\n".join(message_lines))
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ في قراءة الصفقات المفتوحة: {e}")


async def close_all_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        bot_data = context.bot_data
        client = bot_data.get("binance_client")
        testnet = bot_data.get("binance_mode") == "testnet"

        open_trades = bot_data.get("open_trades_data", {})
        if not open_trades:
            await update.message.reply_text("✅ لا توجد صفقات مفتوحة لإغلاقها.")
            return

        closed = []
        for symbol, trade in list(open_trades.items()):
            qty = Decimal(trade.get("quantity", "0"))
            current_price = Decimal(client.get_symbol_ticker(symbol=symbol)["price"])
            value = qty * current_price
            cost = Decimal(trade.get("cost_usdt", "0"))
            profit = value - cost

            bot_data["simulated_usdt_balance"] += value
            bot_data.setdefault("closed_trades", []).append({
                "symbol": symbol,
                "entry_price": trade.get("avg_price"),
                "exit_price": current_price,
                "quantity": qty,
                "buy_time": trade.get("buy_time"),
                "sell_time": time.time(),
                "cost_usdt": cost,
                "profit": profit
            })
            closed.append(f"✅ {symbol} أغلق على {current_price} | ربح: {profit:.2f} USDT")
            del bot_data["open_trades_data"][symbol]

        save_testnet_trades(bot_data)
        await update.message.reply_text("\n".join(closed))
    except Exception as e:
        await update.message.reply_text(f"❌ فشل في غلق الصفقات: {e}")

# --- Main Function --- 
def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("رمز بوت Telegram غير موجود أو غير صالح. الخروج.")
        return
    if not config:
        logger.critical("فشل تحميل ملف الإعدادات. الخروج.")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("switch_mode", switch_mode_command))
    application.add_handler(CommandHandler("balance", get_usdt_balance_command))
    application.add_handler(CommandHandler("open_trades", open_trades_command))
    application.add_handler(CommandHandler("manual_mode_on", manual_mode_on_command))
    application.add_handler(CommandHandler("manual_mode_off", manual_mode_off_command))
    application.add_handler(CommandHandler("auto_mode_on", auto_mode_on_command))
    application.add_handler(CommandHandler("auto_mode_off", auto_mode_off_command))
    application.add_handler(CommandHandler("buy", buy_command))
    application.add_handler(CommandHandler("sell", sell_command)) # Added sell command handler
    application.add_handler(CommandHandler("stop_trading", stop_trading_command))
    application.add_handler(CommandHandler("start_trading", start_trading_command))
    application.add_handler(CommandHandler("profit_summary", profit_summary_command))
    application.add_handler(CommandHandler("open_trades_summary", open_trades_summary_command))
    application.add_handler(CommandHandler("close_trades", close_all_trades_command))
    application.add_handler(CommandHandler("open_trades_summary", open_trades_summary_command))
    application.add_handler(CommandHandler("close_trades", close_all_trades_command))

    # Add message handler for signals (if auto mode is on)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Add handler for unknown commands
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    # Add error handler
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    logger.info("بدء تشغيل البوت...")
    application.job_queue.run_once(lambda *_: asyncio.create_task(monitor_trades_task(application)), when=0)
    application.run_polling()


async def monitor_trades_task(application):
    import asyncio
    import time
    while True:
        try:
            bot_data = application.bot_data
            if bot_data.get("binance_mode") != "testnet":
                await asyncio.sleep(60)
                continue

            client = bot_data.get("binance_client")
            open_trades = bot_data.get("open_trades_data", {})
            trades_to_close = []

            for symbol, trade in open_trades.items():
                avg_price = Decimal(trade.get("avg_price", "0"))
                quantity = Decimal(trade.get("quantity", "0"))
                tp_price = trade.get("tp_price") or avg_price * Decimal("1.01")

                try:
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = Decimal(ticker["price"])
                except Exception:
                    continue

                if current_price >= tp_price:
                    trades_to_close.append((symbol, trade, current_price))

            for symbol, trade, exit_price in trades_to_close:
                quantity = Decimal(trade.get("quantity", "0"))
                cost = Decimal(trade.get("cost_usdt", "0"))
                usdt_gained = quantity * exit_price

                bot_data["simulated_usdt_balance"] += usdt_gained

                closed_trade = {
                    "symbol": symbol,
                    "entry_price": trade.get("avg_price"),
                    "exit_price": exit_price,
                    "quantity": quantity,
                    "buy_time": trade.get("buy_time"),
                    "sell_time": time.time(),
                    "cost_usdt": cost,
                    "profit": usdt_gained - cost
                }

                bot_data["open_trades_data"].pop(symbol, None)
                bot_data.setdefault("closed_trades", []).append(closed_trade)
                save_testnet_trades(bot_data)

                try:
                    await application.bot.send_message(
                        chat_id=bot_data.get("OWNER_ID", 0),
                        text=f"✅ تم جني الأرباح تلقائياً على {symbol} بسعر {exit_price}. 💰 ربح: {closed_trade['profit']:.2f} USDT"
                    )
                except:
                    pass

        except Exception as e:
            print(f"[Monitor TP Error]: {e}")
        await asyncio.sleep(60)


if __name__ == "__main__":
    main()



