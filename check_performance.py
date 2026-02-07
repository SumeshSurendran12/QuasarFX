import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MT5 Configuration
MT5_CONFIG = {
    'login': int(os.getenv('MT5_LOGIN')),
    'password': os.getenv('MT5_PASSWORD'),
    'server': os.getenv('MT5_SERVER'),
}

def check_positions():
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    # Login
    if not mt5.login(MT5_CONFIG['login'], MT5_CONFIG['password'], MT5_CONFIG['server']):
        print("MT5 login failed")
        return

    # Get positions
    positions = mt5.positions_get()
    if positions is None:
        print("No positions found")
        return

    print(f"Current positions ({len(positions)}):")
    total_pnl = 0
    for pos in positions:
        pnl = pos.profit + pos.swap
        total_pnl += pnl
        sl_info = f"SL: {pos.sl:.5f}" if pos.sl > 0 else "SL: None"
        tp_info = f"TP: {pos.tp:.5f}" if pos.tp > 0 else "TP: None"
        print(f"  Ticket: {pos.ticket}, Symbol: {pos.symbol}, Type: {'Buy' if pos.type == 0 else 'Sell'}, "
              f"Volume: {pos.volume}, Price: {pos.price_open}, Current: {pos.price_current}, "
              f"P&L: ${pnl:.2f}, {sl_info}, {tp_info}")

    print(f"Total P&L: ${total_pnl:.2f}")

    # Get account info
    account = mt5.account_info()
    if account:
        print(f"Account Balance: ${account.balance:.2f}, Equity: ${account.equity:.2f}")

    mt5.shutdown()

if __name__ == "__main__":
    check_positions()