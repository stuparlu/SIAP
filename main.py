import asyncio
from pyppeteer import launch


async def main():
    browser = await launch({'headless': False, 'executablePath': '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'})
    page = await browser.newPage()
    # await page.goto('https://www.kickstarter.com/projects/jonathanmann/the-songs-of-adelaide-and-abullah')
    await page.goto('https://www.kickstarter.com/projects/213062534/greeting-from-earth-zgac-arts-capsule-for-et')
    a = input("")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
