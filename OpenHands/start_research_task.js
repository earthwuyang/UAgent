#!/usr/bin/env node
/**
 * OpenHands Research Task Automation
 * 
 * This script uses Puppeteer to:
 * 1. Navigate to localhost:3000
 * 2. Start a conversation
 * 3. Send the research goal about PostgreSQL + pg_duckdb ML-based query routing
 * 4. Monitor progress
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const RESEARCH_GOAL = `research goal: modify postgres and pg_duckdb source code （ to download source code you can utilize the proxy on port localhost:7890, do not use the system-wide postgresql）, first extract pre-opt features from postgres kernel and log to files, then collect dual-execution data (pre-optimization query features that can be found in kernel structures and execution times on dual engine) and train a machine learning model to predict whether postgres engine or duckdb engine executes a query fast and embed the machine learning model into database source code (using the language of the database for example c language) to online route each query to the faster engine, and execute end-to-end experiments to test the ml-based system's performance. A baseline method called threshold-based method should also be implemented, which routes query based on threshold, for example threshold can be 10000 or 50000 or any other value, if postgres estimates the cost of a query is above threshold, then send to duckdb, otherwise send to postgres, and compare the postgres-only, duckdb-only, different threshold-based methods and lightgbm-based method. please record every successful  necessary commands in README.md so that later people can reproduce your results. also record your python packages dependencies in requirements.txt.`;

const SERVER_URL = 'http://localhost:3000';
const LOG_FILE = path.join(__dirname, 'research_automation.log');

// Logging utility
function log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level}] ${message}\n`;
    console.log(logMessage.trim());
    fs.appendFileSync(LOG_FILE, logMessage);
}

// Wait for element with timeout
async function waitForElement(page, selector, timeout = 30000) {
    try {
        await page.waitForSelector(selector, { timeout, visible: true });
        return true;
    } catch (error) {
        log(`Timeout waiting for selector: ${selector}`, 'ERROR');
        return false;
    }
}

// Take screenshot helper
async function takeScreenshot(page, name) {
    const screenshotPath = path.join(__dirname, `screenshot_${name}_${Date.now()}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    log(`Screenshot saved: ${screenshotPath}`);
}

// Main automation function
async function automateResearch() {
    log('='.repeat(80));
    log('OpenHands Research Task Automation Started');
    log('='.repeat(80));

    let browser;
    try {
        // Launch browser
        log('Launching Puppeteer browser...');
        browser = await puppeteer.launch({
            headless: false, // Set to true for headless mode
            defaultViewport: { width: 1920, height: 1080 },
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
            ],
        });

        const page = await browser.newPage();
        
        // Enable console logging from the page
        page.on('console', msg => {
            log(`[BROWSER CONSOLE] ${msg.type()}: ${msg.text()}`, 'DEBUG');
        });

        // Navigate to OpenHands
        log(`Navigating to ${SERVER_URL}...`);
        try {
            await page.goto(SERVER_URL, { 
                waitUntil: 'networkidle2',
                timeout: 60000 
            });
            log('Successfully loaded OpenHands page');
        } catch (error) {
            log(`Failed to navigate to ${SERVER_URL}: ${error.message}`, 'ERROR');
            log('Please ensure the OpenHands server is running on port 3000', 'ERROR');
            await takeScreenshot(page, 'navigation_error');
            throw error;
        }

        await takeScreenshot(page, 'initial_page');

        // Wait for the page to be ready
        log('Waiting for page to be ready...');
        await page.waitForTimeout(3000);

        // Try multiple possible selectors for the chat input
        log('Looking for chat input field...');
        const possibleSelectors = [
            'textarea[placeholder*="Send a message"]',
            'textarea[placeholder*="message"]',
            'input[type="text"][placeholder*="message"]',
            'textarea.chat-input',
            '[data-testid="chat-input"]',
            '[class*="chat"][class*="input"]',
            'textarea',
        ];

        let chatInputFound = false;
        let chatInputSelector = null;

        for (const selector of possibleSelectors) {
            try {
                const element = await page.$(selector);
                if (element) {
                    log(`Found chat input with selector: ${selector}`);
                    chatInputSelector = selector;
                    chatInputFound = true;
                    break;
                }
            } catch (e) {
                // Continue trying other selectors
            }
        }

        if (!chatInputFound) {
            log('Could not find chat input field automatically', 'WARN');
            log('Taking screenshot for manual inspection...');
            await takeScreenshot(page, 'no_input_found');
            
            // Try to list all textareas for debugging
            const textareas = await page.$$('textarea');
            log(`Found ${textareas.length} textarea elements on the page`);
            
            // You may need to manually inspect the page and update the selector
            log('Please inspect the page and update the selector in this script', 'ERROR');
            throw new Error('Chat input field not found');
        }

        // Click on the chat input to focus
        log('Clicking on chat input...');
        await page.click(chatInputSelector);
        await page.waitForTimeout(1000);

        // Type the research goal
        log('Typing research goal...');
        await page.type(chatInputSelector, RESEARCH_GOAL, { delay: 10 });
        log('Research goal typed successfully');

        await takeScreenshot(page, 'message_typed');
        await page.waitForTimeout(2000);

        // Find and click send button
        log('Looking for send button...');
        const sendButtonSelectors = [
            'button[type="submit"]',
            'button[aria-label*="send"]',
            'button[aria-label*="Send"]',
            '[data-testid="send-button"]',
            'button:has-text("Send")',
            'button svg[class*="send"]',
        ];

        let sendButtonFound = false;
        for (const selector of sendButtonSelectors) {
            try {
                const button = await page.$(selector);
                if (button) {
                    log(`Found send button with selector: ${selector}`);
                    await button.click();
                    sendButtonFound = true;
                    log('Message sent successfully!');
                    break;
                }
            } catch (e) {
                // Continue trying
            }
        }

        if (!sendButtonFound) {
            // Try pressing Enter as fallback
            log('Send button not found, trying Enter key...');
            await page.keyboard.press('Enter');
            log('Pressed Enter to send message');
        }

        await takeScreenshot(page, 'message_sent');
        await page.waitForTimeout(3000);

        // Monitor progress
        log('');
        log('='.repeat(80));
        log('Research task submitted! Monitoring progress...');
        log('='.repeat(80));
        log('');

        // Keep the browser open and monitor for a while
        log('The browser will remain open for monitoring.');
        log('You can observe the research progress in real-time.');
        log('');
        log('Research tree visualization should be visible at:');
        log(`  ${SERVER_URL}/research/tree`);
        log('');
        
        // Monitor for 5 minutes, taking periodic screenshots
        const monitorDuration = 5 * 60 * 1000; // 5 minutes
        const screenshotInterval = 30 * 1000; // 30 seconds
        const startTime = Date.now();

        log('Starting monitoring phase (5 minutes)...');
        
        while (Date.now() - startTime < monitorDuration) {
            await page.waitForTimeout(screenshotInterval);
            
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            log(`Monitoring... (${elapsed}s elapsed)`);
            
            await takeScreenshot(page, `progress_${elapsed}s`);
            
            // Try to extract progress information from the page
            try {
                const progressText = await page.evaluate(() => {
                    // Try to find progress indicators
                    const progressElements = document.querySelectorAll('[class*="progress"], [class*="status"]');
                    return Array.from(progressElements)
                        .map(el => el.textContent)
                        .filter(text => text && text.trim())
                        .join(' | ');
                });
                
                if (progressText) {
                    log(`Progress update: ${progressText}`);
                }
            } catch (e) {
                // Ignore extraction errors
            }
        }

        log('');
        log('='.repeat(80));
        log('Monitoring phase complete');
        log('='.repeat(80));
        log('');
        log('The research task continues in the background.');
        log(`Check the full logs at: ${LOG_FILE}`);
        log('');
        
        // Keep browser open for manual inspection
        log('Browser will remain open. Press Ctrl+C to exit.');
        await new Promise(() => {}); // Keep running indefinitely

    } catch (error) {
        log(`Error during automation: ${error.message}`, 'ERROR');
        log(error.stack, 'ERROR');
        
        if (browser) {
            await takeScreenshot(browser.pages()[0], 'error_state');
        }
        
        throw error;
    } finally {
        // Don't close browser automatically to allow manual inspection
        // Uncomment the next line if you want to close automatically
        // if (browser) await browser.close();
    }
}

// Run the automation
if (require.main === module) {
    log('Starting OpenHands Research Automation Script');
    log(`Log file: ${LOG_FILE}`);
    log('');
    
    automateResearch()
        .then(() => {
            log('Automation completed successfully');
        })
        .catch((error) => {
            log('Automation failed', 'ERROR');
            log(error.message, 'ERROR');
            process.exit(1);
        });
}

module.exports = { automateResearch };
