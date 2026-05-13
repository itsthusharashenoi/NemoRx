import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const htmlPath = path.resolve(process.argv[2] || path.join(__dirname, 'filled_prescription.html'));
const pdfPath = path.resolve(process.argv[3] || path.join(__dirname, 'output.pdf'));

const browser = await puppeteer.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

const page = await browser.newPage();
await page.emulateMediaType('print');

const fileUrl = `file://${htmlPath}`;
const viewportWidth = 900;
await page.setViewport({ width: viewportWidth, height: 1100, deviceScaleFactor: 1.5 });
await page.goto(fileUrl, { waitUntil: 'networkidle0', timeout: 45000 });
await new Promise((r) => setTimeout(r, 500));

// Multi-page A4 so print CSS (e.g. page-break-before on audit) produces a separate page for the transcript.
await page.pdf({
  path: pdfPath,
  format: 'A4',
  printBackground: true,
  preferCSSPageSize: true,
  margin: { top: 0, right: 0, bottom: 0, left: 0 },
});

await browser.close();
console.log(`PDF written to ${pdfPath}`);
