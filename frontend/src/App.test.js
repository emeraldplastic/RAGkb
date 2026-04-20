import { render, screen } from '@testing-library/react';
import App from './App';

test('renders auth screen by default', () => {
  localStorage.clear();
  render(<App />);
  const heading = screen.getByText(/welcome back/i);
  expect(heading).toBeInTheDocument();
});
