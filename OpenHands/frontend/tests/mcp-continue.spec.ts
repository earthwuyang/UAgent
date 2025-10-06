import { test, expect } from '@playwright/test';

// Repro script: navigate to a conversation, type 'continue', hit Enter, and capture console/page errors.
test('conversation continue triggers no React 185', async ({ page }, testInfo) => {
  const targetUrl = process.env.TARGET_CONVERSATION_URL || 'http://120.46.207.248:3000/conversations/31f7a7c6fa31456ca4301716105fbd5c';

  const errors: string[] = [];
  const consoles: string[] = [];

  page.on('pageerror', (err) => {
    errors.push(`pageerror: ${err?.message || String(err)}`);
  });
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoles.push(`console.error: ${msg.text()}`);
    }
  });

  await page.goto(targetUrl, { waitUntil: 'domcontentloaded' });

  // Wait for root outlet to ensure the SPA mounted
  await page.locator('#root-outlet').waitFor({ state: 'visible', timeout: 15000 }).catch(() => {});

  // Focus the chat input and type 'continue', then press Enter
  const chatInput = page.getByTestId('chat-input');
  await chatInput.click({ timeout: 15000 });
  await chatInput.type('continue');
  await chatInput.press('Enter');

  // Give the app a moment to process and potentially log errors
  await page.waitForTimeout(2000);

  // Attach artifacts to the test output dir
  const screenshotPath = testInfo.outputPath('conversation-continue.png');
  await page.screenshot({ path: screenshotPath, fullPage: true }).catch(() => {});

  if (errors.length + consoles.length > 0) {
    const combined = [...errors, ...consoles].join('\n');
    testInfo.attach('errors.log', { body: combined, contentType: 'text/plain' });
  }

  // Ensure no React minified 185 error appears
  const hasReact185 = errors.concat(consoles).some((e) => /React error #?185|Maximum update depth exceeded/i.test(e));
  expect(hasReact185).toBeFalsy();
});

