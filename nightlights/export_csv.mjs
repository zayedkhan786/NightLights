import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { Workbook } from '@oai/artifact-tool';

const output = process.argv[2];
const escape = value => {
  if (value === null || value === undefined) return '';
  const text = typeof value === 'boolean' ? (value ? 'True' : 'False') : String(value);
  return /[",\r\n]/.test(text) ? '"'+text.replaceAll('"','""')+'"' : text;
};
const results = [];
for (const selection of ['main_2pixels','context_7pixels']) {
  const input = JSON.parse(await fs.readFile(path.join(output,selection+'.json'),'utf8'));
  const wb = Workbook.create();
  const sheet = wb.worksheets.add('Daily measurements');
  const range = sheet.getRangeByIndexes(0,0,input.rows.length+1,input.columns.length);
  range.values = [input.columns,...input.rows];
  sheet.getRange('A1:F6').format.font = {name:'Arial',size:11};
  sheet.getRange('A1:F6').format.columnWidthPx = 145;
  sheet.getRange('A1:F1').format.wrapText = true;
  sheet.getRange('A1:F1').format.rowHeightPx = 42;
  sheet.getRange('A1:F1').format.fill = '#263747';
  sheet.getRange('A1:F1').format.font = {name:'Arial',size:11,color:'#FFFFFF',bold:true};
  sheet.getRange('B2:F6').setNumberFormat('0.000');
  wb.recalculate();
  const values = range.values;
  assert.deepEqual(values,[input.columns,...input.rows]);
  console.log((await wb.inspect({kind:'region',sheetId:sheet.name,range:'A1:F4',maxChars:1000})).ndjson);
  const preview = await wb.render({sheetName:sheet.name,range:'A1:F6',scale:1.5,format:'png'});
  await fs.writeFile(path.join(output,selection+'_preview.png'),new Uint8Array(await preview.arrayBuffer()));
  const target = path.join(output,'miami_'+selection+'.csv');
  await fs.writeFile(target,values.map(row=>row.map(escape).join(',')).join('\r\n')+'\r\n','utf8');
  results.push({selection,file:path.basename(target),rows:input.rows.length,columns:input.columns.length,artifact_values_verified:true});
  console.log('Saved',target);
}
await fs.writeFile(path.join(output,'artifact_verification.json'),JSON.stringify(results,null,2));
