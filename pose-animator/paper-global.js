const paperRef = window.paper;

if (!paperRef) {
  throw new Error(
    'Paper.js global is missing. Make sure vendor/paper-full.min.js is loaded before module scripts.'
  );
}

export default paperRef;
