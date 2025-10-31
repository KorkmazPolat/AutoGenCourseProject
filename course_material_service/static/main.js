document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById('course-form');
  const overlay = document.getElementById('loading-overlay');
  const bar = document.getElementById('progress-bar');
  const pct = document.getElementById('progress-text');
  const hint = document.getElementById('progress-hint');

  form?.addEventListener('submit', async (e) => {
    e.preventDefault();
    overlay.classList.remove('hidden');

    let progress = 0;
    const milestones = [15, 35, 55, 75, 90];
    const hints = ['Drafting plan…', 'Outlining slides…', 'Writing narration…', 'Rendering visuals…', 'Finalizing output…'];
    let step = 0;

    const interval = setInterval(() => {
      if(step < milestones.length && progress < milestones[step]){
        progress += Math.max(1, Math.round((milestones[step]-progress)/5));
      } else if(step < milestones.length){
        step++;
        hint.textContent = hints[Math.min(step, hints.length-1)];
      } else if(progress < 95){
        progress++;
      }
      bar.style.width = progress + '%';
      pct.textContent = progress + '%';
    }, 250);

    const fd = new FormData(form);
    const action = e.submitter.getAttribute('formaction') || form.getAttribute('action');

    try {
      const response = await fetch(action, { method:'POST', body:fd, credentials:'same-origin' });
      const html = await response.text();
      progress = 100; bar.style.width='100%'; pct.textContent='100%';
      clearInterval(interval);

      const values = {};
      for(const el of form.elements) if(el.name) values[el.name]=el.value;

      document.open();
      document.write(html);
      document.close();

      const newForm = document.getElementById('course-form');
      for(const el of newForm.elements) if(el.name && values[el.name]) el.value = values[el.name];

    } catch(err){
      clearInterval(interval);
      hint.textContent='Something went wrong. Please try again.';
      hint.classList.add('text-red-600');
    }
  });
});
